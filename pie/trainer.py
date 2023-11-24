from time import time

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
        initial_iter_loop = self.config.step % self.config.total_train_step_per_epoch
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.train_loader,
            initial=initial_iter_loop,
            total=self.config.total_train_step_per_epoch,
            disable=(self.rank != 0),
            ascii=True,
        )
        for i, (inputs, _, target) in enumerate(loop):
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
                (i == 20 and self.config.debug)
                or (i == self.config.total_train_step_per_epoch + 1 - initial_iter_loop)
            ):
                loop.close()
                print(
                    f"Max memory allocated: \
                    {torch.cuda.max_memory_allocated(device=self.device)/(1024**3)}"
                )
                break

            self.config.step += 1

        return list_loss

    def eval_loop(self) -> float:
        self.metric.reset()
        self.model.eval()
        loop: tqdm[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = tqdm(
            self.valid_loader, disable=(self.rank != 0), ascii=True
        )

        with torch.no_grad():
            for i, (inp_ids, mask, targets) in enumerate(loop):
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

                loop.set_postfix(
                    average_perplexity=avg_perplexiy.item(),
                    perplexity=score.item(),
                )

                if i == 20 and self.config.debug:
                    loop.close()
                    break

            perplexity = self.metric.compute().item()
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

        if self.config.checkpoint:
            save_checkpoints(self.config, self.model, self.optimizer, self.lr_scheduler)

        return self.model
