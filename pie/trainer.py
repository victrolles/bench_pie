from time import time
import idr_torch
from statistics import mean
import csv
import torch.distributed as dist

import mlflow
import torch
from torch import profiler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import TrainingConfig
from .utils import PieModel, print_rank_0, save_checkpoints


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: PieModel,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        visu_data: tuple[list[str], BatchEncoding, list[str]],
        criterion: _Loss,
        metric: Metric,
        optimizer: Optimizer,
        lr_scheduler: LambdaLR | None,
        device: torch.device | str = "cuda",
        rank: int = 0,
        training_dist: str = "deepspeed",
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.visu_data = visu_data

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = criterion
        self.metric = metric

        self.device = device
        self.rank = rank
        self.training_dist = training_dist

        self.config = config

    @property
    def epoch(self) -> int:
        return self.config.step // self.config.total_train_step_per_epoch + 1

    @torch.no_grad()
    def generate(self, inputs: BatchEncoding, generate_kwargs=None) -> list[list[str]]:
        """returns a list of generated texts"""

        inputs = inputs.to(self.device)

        self.model.forward(input_ids=inputs["input_ids"])  # dummy forward for FSDP
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        input_tokens_lengths = [x.shape[0] for x in inputs["input_ids"]]
        output_tokens_lengths = [x.shape[0] for x in outputs]

        total_new_tokens = [
            o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
        ]
        # outputs_full = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        outputs_new = []
        for i, total_new_token in enumerate(total_new_tokens):
            outputs_new.append(
                self.tokenizer.batch_decode(
                    [outputs[i][-total_new_token:]], skip_special_tokens=True
                )[0]
            )

        return outputs_new

    def visual_test(
        self,
        inp_prompt: list[str],
        inp_tensor: BatchEncoding,
        label_prompt: list[str],
    ) -> None:
        """Prints generated text and target text"""
        generated = self.generate(inp_tensor)

        for prompt, gen, lab in zip(inp_prompt, generated, label_prompt):
            print_rank_0(f"Prompt:\n{prompt}\n")
            print_rank_0(f"Generated:\n{gen}\n")
            print_rank_0(f"Target:\n{lab}\n")
            print_rank_0("******************************")

        # Add step number to prompt as a visual check in mlflow
        inp_prompt = [f"Step {self.config.step}: " + prompt for prompt in inp_prompt]

        if self.config.track:
            # May change in future version of mlflow
            mlflow.llm.log_predictions(inp_prompt, generated, label_prompt)

    def prepare_for_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize logits and target Tensors for pytorch CrossEntropyLoss"""
        batch_size, seq_length, vocab_size = logits.shape
        logits = logits.view(batch_size * seq_length, vocab_size)
        target = target.view(batch_size * seq_length)
        return logits, target

    def train_loop(self) -> torch.Tensor:
        self.model.train()
        list_loss = torch.Tensor([]).to(self.device)
        train_list_time_iter = []
        initial_iter_loop = self.config.step % self.config.total_train_step_per_epoch
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.train_loader,
            initial=initial_iter_loop,
            total=self.config.total_train_step_per_epoch,
            disable=(self.rank != 0),
            ascii=True,
        )
        for i, (inputs, _, target) in enumerate(loop):
            
            start_time_iter = time()
            if i == 0:
                inputs_shape = inputs.shape

            inputs = inputs.to(self.device)
            target = target.to(self.device)
            out = self.model(input_ids=inputs)

            logits, target = self.prepare_for_loss(out["logits"], target)
            loss: torch.Tensor = self.criterion(logits, target)

            if self.training_dist == "deepspeed":
                self.model.backward(loss)
                self.model.step()

            else:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            train_list_time_iter.append(time() - start_time_iter)

            # To monitor training
            list_loss = torch.cat((list_loss, loss.detach().data.view(1)))
            avg_loss = list_loss.mean().item()
            # TODO: check if it doesn't slow training
            loop.set_postfix(
                average_loss=avg_loss,
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]["lr"],
            )

            if self.config.track:
                mlflow.log_metrics(
                    {
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                    },
                    step=self.config.step,
                )

            if self.config.profile:
                self.prof.step()

            if (
                (i == 75 and self.config.debug)
                or (i == self.config.total_train_step_per_epoch + 1 - initial_iter_loop)
            ):
                loop.close()
                print(
                    f"Max memory allocated: \
                    {torch.cuda.max_memory_allocated(device=self.device)/(1024**3)}"
                )
                break

            self.config.step += 1

        # tokens/second/gpu = (seq/iter/gpu) * (tokens/seq) / (iter/second)
        train_tokens_s = inputs_shape[0] * inputs_shape[1] / mean(train_list_time_iter[:25])
        print(f"Train token per second of GPU n°{idr_torch.rank}: {train_tokens_s:.1f}")
        tensor_train_tokens_s = torch.tensor(train_tokens_s, device=self.device)
        # Sum the throughput of all GPUs
        dist.all_reduce(tensor_train_tokens_s, op=dist.ReduceOp.SUM)
        self.train_tokens_s = tensor_train_tokens_s.item()

        return list_loss

    def eval_loop(self) -> float:
        self.metric.reset()
        self.model.eval()
        list_tokens_s_per_gpu = []
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.valid_loader, disable=(self.rank != 0), ascii=True
        )

        with torch.no_grad():
            for i, (inp_ids, mask, targets) in enumerate(loop):

                start_time_iter = time()

                inp_ids = inp_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                preds: CausalLMOutputWithPast = self.model(
                    input_ids=inp_ids, attention_mask=mask
                )

                score: torch.Tensor = self.metric(
                    preds.logits.to(torch.float32), targets
                )
                avg_perplexiy: torch.Tensor = self.metric.compute()

                list_tokens_s_per_gpu.append(inp_ids.shape[0] * inp_ids.shape[1] / (time() - start_time_iter))

                loop.set_postfix(
                    average_perplexity=avg_perplexiy.item(),
                    perplexity=score.item(),
                )

                if i == 75 and self.config.debug:
                    loop.close()
                    break

            perplexity = self.metric.compute().item()

            val_tokens_s = mean(list_tokens_s_per_gpu[:25])
            print(f"Inference token per second of GPU n°{idr_torch.rank}: {val_tokens_s:.1f}")
            tensor_val_tokens_s = torch.tensor(val_tokens_s, device=self.device)
            dist.all_reduce(tensor_val_tokens_s, op=dist.ReduceOp.SUM)
            self.val_tokens_s = tensor_val_tokens_s.item()

            if self.config.track:
                mlflow.log_metric("perplexity", perplexity, step=self.config.step)

        return perplexity

    def train(self, epochs: int) -> PieModel:
        inp_prompt, inp_tensor, label_prompt = self.visu_data
        self.visual_test(inp_prompt, inp_tensor, label_prompt)

        perplexity = self.eval_loop()

        for epoch in range(epochs - self.epoch + 1):
            start_epoch = time()
            print_rank_0(
                f"#################### Epoch {self.epoch}/{epochs} ####################"
            )

            # TODO: find a cleaner way to use profiler (with a decorator ?)
            if self.config.profile:
                with profiler.profile(
                    schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
                    on_trace_ready=profiler.tensorboard_trace_handler(
                        str(self.config.profiler_path)
                    ),
                    profile_memory=True,
                    with_stack=False,
                    record_shapes=False,
                ) as prof:
                    self.prof = prof
                    list_loss = self.train_loop()

            else:
                list_loss = self.train_loop()

            perplexity = self.eval_loop()
            self.visual_test(inp_prompt, inp_tensor, label_prompt)

            print_rank_0(
                f"Epoch duration: {(time() - start_epoch):.2f}s |",
                f"average loss: {list_loss.mean().item():.3f} |",
                f"perplexity: {perplexity:.3f}\n"
            )
            print_rank_0("#" * 60, "\n")

        if idr_torch.rank==0:

            print(
                f"\n------------------- results benckmarks -------------------",
                f"\nGPU type : {torch.cuda.get_device_name(self.device)}",
                f"\nGPU number : {idr_torch.size}",
                f"\nModel name : {self.config.model_name}",
                f"\nOptimized DDP : {self.config.training_dist}",
                f"\nDeepspeed ZeRO stage: {self.config.stage}",
                f"\nTraining batch size : {self.config.batch_size}",
                f"\nInference batch size : {self.config.valid_batch_size}",
                f"\nSequence length: {self.config.seq_length}",
                f"\nNumber of layer freezed : {self.config.nb_layer_freezed}",
                f"\nEpoch duration : {(time() - start_epoch):.2f}s",
                f"\nGlobal Training Token/s : {self.train_tokens_s}",
                f"\nGlobal Inference Token/s : {self.val_tokens_s}",
                f"\nGlobal Max Memory Allocated : {round(torch.cuda.max_memory_allocated(self.device)/2**30, 2):.2f}",
                f"\nGlobal Max Memory Reserved : {round(torch.cuda.max_memory_reserved(self.device)/2**30, 2):.2f}",
                f"\nAverage loss : {list_loss.mean().item():.3f}",
                f"\nPerplexity : {perplexity}",
                f"\n----------------------------------------------------------\n")

            with open("../../bench_pie.csv",'a') as file:
                writer=csv.writer(file)
                # typeGPU,nbrGPU,model_name,training_dist,stage,batch_size,valid_batch_size,seq_length,nb_layer_freezed,epoch_duration,train_tokens_s,val_tokens_s,maxMemAlloc,maxMemRes,mean_loss,perplexity
                writer.writerow([torch.cuda.get_device_name(self.device),
                                idr_torch.size,
                                self.config.model_name,
                                self.config.training_dist,
                                self.config.stage,
                                self.config.batch_size,
                                self.config.valid_batch_size,
                                self.config.seq_length,
                                self.config.nb_layer_freezed,
                                (time() - start_epoch),
                                self.train_tokens_s,
                                self.val_tokens_s,
                                round(torch.cuda.max_memory_allocated(self.device)/2**30, 2),
                                round(torch.cuda.max_memory_reserved(self.device)/2**30, 2),
                                list_loss.mean().item(),
                                perplexity])   

        if self.config.checkpoint:
            save_checkpoints(self.config, self.model, self.optimizer, self.lr_scheduler)

        return self.model
