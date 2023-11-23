import os

import idr_torch
import mlflow
import torch
import torch.distributed as dist
from torchmetrics.text import Perplexity
from transformers import AutoTokenizer

from .config import TrainingConfig
from .data_pipeline import get_dataloaders
from .modeling import modeling
from .trainer import Trainer
from .utils import load_checkpoints, print_rank_0  # , save_model

DEVICE = torch.device("cuda", idr_torch.local_rank)

torch.manual_seed(53)
torch.cuda.manual_seed(53)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # TODO: check if true is better


def train(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # For test we use padding so we need to set the padding token
    # It doesn't really matter what it is
    tokenizer.pad_token_id = config.pad_token_id
    tokenizer.padding_side = "left"
    assert tokenizer.bos_token_id != config.pad_token_id
    assert tokenizer.eos_token_id != config.pad_token_id

    train_loader, valid_loader, visu_data = get_dataloaders(config, tokenizer)
    model, optimizer, lr_scheduler = modeling(config)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    metric = Perplexity(ignore_index=config.pad_token_id).to(DEVICE)

    if config.step > 0:
        model, optimizer, lr_scheduler = load_checkpoints(
            config, model, optimizer, lr_scheduler
        )

    # Assure that every process has the same config.run_name
    if idr_torch.world_size > 1:
        dist.barrier()
        objects = [config.run_name]
        dist.broadcast_object_list(objects, src=0, device=DEVICE)
        config.run_name = objects[0]

    print_rank_0(f"Start training with the following config:\n{config}")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        visu_data=visu_data,
        criterion=criterion,
        metric=metric,
        config=config,
        device=DEVICE,
        rank=idr_torch.rank,
        training_dist=config.training_dist,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    if config.track and idr_torch.rank == 0:
        mlflow.set_tracking_uri("file://" + str(config.mlflow_path))
        mlflow.set_experiment(config.exp_name)

        if mlflow.search_runs(filter_string=f"run_name='{config.run_name}'").empty:
            run_id = None
        else:
            run_id = mlflow.search_runs(
                filter_string=f"run_name='{config.run_name}'"
            ).iloc[0]["run_id"]

        with mlflow.start_run(run_name=config.run_name, run_id=run_id) as _:
            mlflow.log_params(config.export())
            model = trainer.train(config.epochs)

    else:
        config.track = False
        model = trainer.train(config.epochs)

    # save model if not in debug mode and the training is finished
    # if not config.debug and model is not None:
    #     save_model(config, model)
