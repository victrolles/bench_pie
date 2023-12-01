import functools
import re
from time import time
from typing import Any

import deepspeed
import idr_torch
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .config import TrainingConfig
from .optimizer import get_optimizer_scheduler
from .utils import PieModel, TransformerModel, print_rank_0

DEVICE = torch.device("cuda", idr_torch.local_rank)


def get_hf_model(config: TrainingConfig) -> TransformerModel:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    if config.nb_layer_freezed > 0 and config.training_dist != "fsdp":
        for name, params in model.named_parameters():
            if re.search(r"embed", name) is not None:
                params.requires_grad = False
            elif re.search(r"\.(\d+)\.", name) is not None:
                if (
                    int(re.search(r"\.(\d+)\.", name).group(1))
                    < config.nb_layer_freezed
                ):
                    params.requires_grad = False

    elif config.nb_layer_freezed > 0 and config.training_dist == "fsdp":
        print_rank_0(
            "Warning: Freezing layers is not supported with FSDP,",
            "not freezing any layer",
        )

    return model


def get_fsdp_model(config: TrainingConfig) -> (FSDP, Optimizer, LambdaLR | None):
    dist.init_process_group(
        "nccl", rank=idr_torch.rank, world_size=idr_torch.world_size
    )
    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=100
    # )  # TODO: To try
    torch.cuda.set_device(DEVICE)

    if "llama" in config.model_name.lower() or "falcon" in config.model_name.lower():
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer
                if "llama" in config.model_name.lower()
                else FalconDecoderLayer
            },
        )
    else:
        print_rank_0("No auto wrap policy: FSDP is not optimized for this model")
        auto_wrap_policy = None

    model = get_hf_model(config).to(DEVICE)
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    optimizer, lr_scheduler = get_optimizer_scheduler(config, model)

    return model, optimizer, lr_scheduler


def get_ds_config(config: TrainingConfig) -> dict[str, Any]:
    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": config.stage,
        },
        "zero_allow_untested_optimizer": True,
    }

    return ds_config


def get_ds_model(
    config: TrainingConfig,
) -> (deepspeed.DeepSpeedEngine, Optimizer, LambdaLR | None):
    start_model = time()  # Measure time

    deepspeed.init_distributed(
        dist_backend="nccl",
        init_method="env://",
        distributed_port = idr_torch.master_port
    )

    ds_config = get_ds_config(config)

    _ = TrainingArguments(output_dir="./", deepspeed=ds_config)
    model = get_hf_model(config)
    optimizer, lr_scheduler = get_optimizer_scheduler(config, model)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )

    # print_trainable_parameters(model)  # TODO: Make it work with DS model
    print_rank_0(f"Initialized Model and Tokenizer in {time() - start_model:.3f}s")

    return model_engine, optimizer, lr_scheduler


def modeling(config: TrainingConfig) -> (PieModel, Optimizer, LambdaLR | None):
    if config.training_dist == "deepspeed":
        return get_ds_model(config)
    elif config.training_dist == "fsdp":
        return get_fsdp_model(config)
    else:
        raise ValueError(f"Unknown distribution type: {config.training_dist}")
