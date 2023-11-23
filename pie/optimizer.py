from functools import partial
from math import cos, pi
from typing import TypeAlias

import deepspeed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainingConfig

PieModel: TypeAlias = deepspeed.DeepSpeedEngine | FSDP


def _linear_cosine_with_warmup(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    anneal_strategy: str,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    elif anneal_strategy == "linear":
        return max(
            0.0,
            (
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps))
            ),
        )

    elif anneal_strategy == "cosine":
        return max(
            0.0,
            (
                cos(
                    (
                        float(max(0, current_step - num_warmup_steps))
                        / float(max(1, num_training_steps - num_warmup_steps))
                    )
                    * pi
                )
                + 1
            )
            / 2,
        )

    else:
        raise ValueError(f"Unknown anneal strategy (lr scheduler): {anneal_strategy}")


def get_scheduler(config: TrainingConfig, optimizer: Optimizer) -> LambdaLR:
    lr_lambda = partial(
        _linear_cosine_with_warmup,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.total_train_step,
        anneal_strategy=config.lr_scheduler,
    )
    return LambdaLR(optimizer, lr_lambda, -1)


def get_optimizer_scheduler(
    config: TrainingConfig, model: PieModel
) -> (Optimizer, LambdaLR | None):
    match config.optimizer:
        case "adamw":
            optimizer = AdamW(
                params=model.parameters(),
                lr=config.lr,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.adam_weight_decay,
            )
        case "sgd":
            optimizer = SGD(
                params=model.parameters(),
                lr=config.lr,
                momentum=config.sgd_momentum,
                weight_decay=config.sgd_weight_decay,
            )
        case _:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    if config.lr_scheduler is not None:
        lr_scheduler = get_scheduler(config, optimizer)
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler
