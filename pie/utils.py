from typing import Protocol, TypeAlias

import deepspeed
import idr_torch
import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel

from .config import TrainingConfig


class _InnerTransformerModel(Protocol):
    word_embeddings: Module
    h: list[Module]


class _TransformerModel(Protocol):
    transformer: _InnerTransformerModel


class TransformerModel(PreTrainedModel, _TransformerModel):
    pass


PieModel: TypeAlias = deepspeed.DeepSpeedEngine | FSDP


def print_rank_0(*args, **kwargs):
    """Print only on rank 0"""
    if idr_torch.rank == 0:
        print(*args, **kwargs)


def save_checkpoints(
    config: TrainingConfig,
    model: PieModel,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR | None,
):
    """Save model and optimizer state dict for checkpointing"""
    print_rank_0("Saving checkpoint...")

    if not config.checkpoint_path.exists() and idr_torch.rank == 0:
        config.checkpoint_path.mkdir(parents=True)

    if config.training_dist == "fsdp":
        # We keep the default save policy for the model
        save_policy = ShardedStateDictConfig()

        with FSDP.state_dict_type(
            model,
            # We use the sharded state dict because we're gonna load the sharded model
            # and optimizer directly. It's faster like that
            StateDictType.SHARDED_STATE_DICT,
            save_policy,
        ):
            # every process need to do the following because we save
            # the sharded state dict of the model and optimizer
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)

            torch.save(
                model_state,
                config.sharded_model_checkpoint_path,
            )
            torch.save(optim_state, config.sharded_optimizer_checkpoint_path)

    elif config.training_dist == "deepspeed":
        # We simply use the deepspeed save_checkpoint method
        model.save_checkpoint(config.checkpoint_path)

    else:
        # Here it's the classic way to save the model and optimizer with pytorch
        # We do it only on rank 0 in case we use DDP with qlora or else in the future
        if idr_torch.rank == 0:
            torch.save(model.state_dict(), config.model_checkpoint_path)
            torch.save(optimizer.state_dict(), config.optimizer_checkpoint_path)

    if (
        config.lr_scheduler is not None
        and config.training_dist != "deepspeed"
        and idr_torch.rank == 0
    ):
        torch.save(lr_scheduler.state_dict(), config.lr_scheduler_checkpoint_path)

    print_rank_0("Checkpoint saved.")


def load_checkpoints(
    config: TrainingConfig,
    model: PieModel,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR | None,
):
    """Load model and optimizer state dict from a checkpointing directory"""
    print_rank_0("Loading checkpoint...")

    if config.training_dist == "fsdp":
        # We load the sharded state dict we saved before directly on the gpus
        load_policy = ShardedStateDictConfig()
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, load_policy):
            model_state = torch.load(config.sharded_model_checkpoint_path)
            optim_state = torch.load(config.sharded_optimizer_checkpoint_path)

            model.load_state_dict(model_state)

            optim_state = FSDP.optim_state_dict_to_load(optim_state, model, optimizer)
            optimizer.load_state_dict(optim_state)

    elif config.training_dist == "deepspeed":
        # We simply use the deepspeed load_checkpoint method
        model.load_checkpoint(config.checkpoint_path)

    else:
        # Here it's the classic way to load the model and optimizer with pytorch
        model.load_state_dict(torch.load(config.model_checkpoint_path))
        optimizer.load_state_dict(torch.load(config.optimizer_checkpoint_path))

    if config.lr_scheduler is not None and config.training_dist != "deepspeed":
        lr_scheduler.load_state_dict(torch.load(config.lr_scheduler_checkpoint_path))

    print_rank_0("Checkpoint loaded.")
    return model, optimizer, lr_scheduler


def save_model(config: TrainingConfig, model: PieModel):
    """Save only model state dict"""
    print_rank_0("Saving model...")

    if not config.save_path.exists() and idr_torch.rank == 0:
        config.save_path.mkdir(parents=True)

    if config.training_dist == "fsdp":
        # We use the full state dict because we're gonna save the full model on one file
        # offload_to_cpu=True and rank0_only=True make the saving faster and save memory
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            # We need to call state_dict() on every process to get every parameters
            model_state = model.state_dict()

            # We save the model on rank 0 only
            if idr_torch.rank == 0:
                torch.save(model_state, config.save_path / "model.pt")

    elif config.training_dist == "deepspeed":
        # TODO: Find a way to only save the model with deepspeed
        model.save_checkpoint(config.save_path)

    else:
        # Here it's the classic way to save the model with pytorch
        if idr_torch.rank == 0:
            torch.save(model.state_dict(), config.save_path / "model.pt")

    print_rank_0("Model saved.")
