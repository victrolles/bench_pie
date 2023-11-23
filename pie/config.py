#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, TypeVar

import idr_torch

T = TypeVar("T", bound="TrainingConfig")


@dataclass(kw_only=True)
class TrainingConfig:
    config_file: str | None = field(
        default=None, metadata={"converter": str, "export": False}
    )
    model_dir: Path = field(
        default=Path.cwd() / "model", metadata={"converter": Path, "export": False}
    )
    checkpoints_dir: Path = field(
        default=Path.cwd() / "checkpoints",
        metadata={"converter": Path, "export": False},
    )
    csv_data_path: Path = field(
        default=Path.cwd() / "dataset" / "train.csv",
        metadata={"converter": Path, "export": False},
    )
    profiler_path: Path = field(
        default=Path.cwd() / "profiler", metadata={"converter": Path, "export": False}
    )
    mlflow_path: Path = field(
        default=Path.cwd() / "mlruns", metadata={"converter": Path, "export": False}
    )
    exp_name: str = field(
        default="PIE-UNK", metadata={"converter": str, "export": False}
    )
    run_name: str = field(default="", metadata={"converter": str, "export": False})
    nb_tokens: int = field(default=4345856, metadata={"converter": int, "export": True})
    seq_length: int = field(default=2048, metadata={"converter": int, "export": True})
    pad_token_id: int = field(default=0, metadata={"converter": int, "export": True})
    model_name: str = field(
        default="meta-llama/Llama-2-13b-hf", metadata={"converter": str, "export": True}
    )
    step: int = field(default=0, metadata={"converter": int, "export": False})
    epochs: int = field(default=2, metadata={"converter": int, "export": True})
    lr: float = field(default=1e-05, metadata={"converter": float, "export": True})
    batch_size: int = field(default=4, metadata={"converter": int, "export": True})
    valid_batch_size: int = field(
        default=4, metadata={"converter": int, "export": True}
    )
    stage: int = field(default=1, metadata={"converter": int, "export": True})
    nb_layer_freezed: int = field(
        default=0, metadata={"converter": int, "export": True}
    )
    training_dist: str = field(
        default="deepspeed", metadata={"converter": str, "export": True}
    )
    debug: bool = field(default=False, metadata={"converter": bool, "export": False})
    profile: bool = field(default=False, metadata={"converter": bool, "export": False})
    track: bool = field(default=False, metadata={"converter": bool, "export": False})
    checkpoint: bool = field(
        default=False, metadata={"converter": bool, "export": False}
    )
    optimizer: str = field(default="adamw", metadata={"converter": str, "export": True})
    adam_beta1: float = field(
        default=0.9, metadata={"converter": float, "export": True}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"converter": float, "export": True}
    )
    adam_eps: float = field(
        default=1e-08, metadata={"converter": float, "export": True}
    )
    adam_weight_decay: float = field(
        default=1e-2, metadata={"converter": float, "export": True}
    )
    sgd_momentum: float = field(
        default=0.0, metadata={"converter": float, "export": True}
    )
    sgd_weight_decay: float = field(
        default=0.0, metadata={"converter": float, "export": True}
    )
    lion_beta1: float = field(
        default=0.9, metadata={"converter": float, "export": True}
    )
    lion_beta2: float = field(
        default=0.99, metadata={"converter": float, "export": True}
    )
    lion_weight_decay: float = field(
        default=0.0, metadata={"converter": float, "export": True}
    )
    lion_optim_bits: int = field(
        default=32, metadata={"converter": int, "export": True}
    )
    lion_is_paged: bool = field(
        default=False, metadata={"converter": bool, "export": True}
    )
    num_warmup_steps: int = field(
        default=20, metadata={"converter": int, "export": False}
    )
    lr_scheduler: str | None = field(
        default=None, metadata={"converter": str, "export": True}
    )

    def __post_init__(self) -> None:
        for dataclass_field in fields(self):
            converter: Callable[[Any], Any] | None = dataclass_field.metadata.get(
                "converter", None
            )
            if converter is not None:
                value = getattr(self, dataclass_field.name)
                if value is not None:
                    self.__setattr__(dataclass_field.name, converter(value))

        if self.run_name == "":
            self.run_name = f"{self.exp_name.lower()}-{time.time_ns()}"

    @classmethod
    def from_mappings(cls: type[T], *mappings: Mapping[str, Any] | Namespace) -> T:
        unified_dict: dict[str, Any] = dict()
        for mapping in mappings:
            if isinstance(mapping, Namespace):
                mapping = vars(mapping)
            unified_dict.update(**mapping)
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in unified_dict.items() if k in class_fields})

    def export(self, *, full: bool = False) -> dict[str, Any]:
        all_fields = fields(self)
        all_exportable_fields = [f.name for f in all_fields if f.metadata.get("export")]
        filtered_export = dict()
        for key, value in vars(self).items():
            if full or key in all_exportable_fields:
                filtered_export[key] = (
                    value if not isinstance(value, Path) else str(value)
                )
        return filtered_export

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.export(full=True), indent=4))
        print(f"Config dumped to {path}")

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        return cls.from_mappings(json.loads(path.read_text()))

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--model-dir", "--model_dir", dest="model_dir")
        parser.add_argument(
            "--checkpoints-dir", "--checkpoints_dir", dest="checkpoints_dir"
        )
        parser.add_argument(
            "--csv-data-path",
            "--csv_data_path",
            dest="csv_data_path",
        )
        parser.add_argument("--profiler-path", "--profiler_path", dest="profiler_path")
        parser.add_argument("--mlflow_path", "--mlflow-path", dest="mlflow_path")
        parser.add_argument("--exp-name", "--exp_name", dest="exp_name")
        parser.add_argument("--run-name", "--run_name", dest="run_name")
        parser.add_argument(
            "--sequence-length",
            "--sequence_length",
            "--seq-length",
            "--seq_length",
            dest="seq_length",
            help="Length of the sequence",
        )
        parser.add_argument("--pad-token-id", "--pad_token_id", dest="pad_token_id")
        parser.add_argument("--model-name", "--model_name", dest="model_name")
        parser.add_argument("--step")
        parser.add_argument("--epochs")
        parser.add_argument("--lr")
        parser.add_argument("--bsz", "--batch-size", "--batch_size", dest="batch_size")
        parser.add_argument(
            "--valid-bsz",
            "--valid-batch-size",
            "--valid_batch_size",
            dest="valid_batch_size",
        )
        parser.add_argument("--stage", type=int, choices=range(4))
        parser.add_argument(
            "--nb-layer-freezed",
            "--nb_layer_freezed",
            dest="nb_layer_freezed",
        )
        parser.add_argument(
            "--dist",
            "--training-dist",
            "--training_dist",
            dest="training_dist",
            help="'fsdp', deepspeed'",
            choices=["fsdp", "deepspeed"],
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--track", help="activate mlflow tracking", action="store_true"
        )
        parser.add_argument("--checkpoint", action="store_true")
        parser.add_argument(
            "--optimizer",
            help="'adamw', 'sgd' or 'lion'",
            choices=["adamw", "sgd", "lion"],
        )
        parser.add_argument("--adam-beta1", "--adam_beta1", dest="adam_beta1")
        parser.add_argument("--adam-beta2", "--adam_beta2", dest="adam_beta2")
        parser.add_argument("--adam-eps", "--adam_eps", dest="adam_eps")
        parser.add_argument(
            "--adam-weight-decay", "--adam_weight_decay", dest="adam_weight_decay"
        )
        parser.add_argument("--sgd-momentum", "--sgd_momentum", dest="sgd_momentum")
        parser.add_argument(
            "--sgd-weight-decay", "--sgd_weight_decay", dest="sgd_weight_decay"
        )
        parser.add_argument("--lion-beta1", "--lion_beta1", dest="lion_beta1")
        parser.add_argument("--lion-beta2", "--lion_beta2", dest="lion_beta2")
        parser.add_argument(
            "--lion-weight-decay", "--lion_weight_decay", dest="lion_weight_decay"
        )
        parser.add_argument(
            "--lion-optim-bits", "--lion_optim_bits", dest="lion_optim_bits"
        )
        parser.add_argument("--lion-is-paged", "--lion_is_paged", dest="lion_is_paged")
        parser.add_argument(
            "--num-warmup-steps",
            "--num_warmup_steps",
            dest="num_warmup_steps",
        )
        parser.add_argument(
            "--lr-scheduler",
            "--lr_scheduler",
            help="'linear' or 'cosine'",
            choices=["linear", "cosine"],
        )

        return parser

    @property
    def data_groups(self) -> list[str]:
        if self.ticket_groups == "both":
            data_groups = ["e-support", "gestutil"]
        else:
            data_groups = [self.ticket_groups]

        if self.include_web:
            data_groups.append("web")

        return data_groups

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name

    @property
    def save_path(self) -> Path:
        return self.checkpoints_dir / self.run_name

    @property
    def checkpoint_path(self) -> Path:
        return self.save_path / f"step_{self.step}"

    @property
    def sharded_model_checkpoint_path(self) -> Path:
        return (
            self.checkpoint_path
            / f"model_state-step_{self.step}-rank_{idr_torch.rank}.pt"
        )

    @property
    def model_checkpoint_path(self) -> Path:
        return self.checkpoint_path / f"model_state-step_{self.step}.pt"

    @property
    def sharded_optimizer_checkpoint_path(self) -> Path:
        return (
            self.checkpoint_path
            / f"optim_state-step_{self.step}-rank_{idr_torch.rank}.pt"
        )

    @property
    def optimizer_checkpoint_path(self) -> Path:
        return self.checkpoint_path / f"optim_state-step_{self.step}.pt"

    @property
    def lr_scheduler_checkpoint_path(self) -> Path:
        return self.checkpoint_path / f"lr_scheduler_state-step_{self.step}.pt"

    @property
    def config_checkpoint_path(self) -> Path:
        return self.checkpoint_path / f"config-step_{self.step}.json"

    @property
    def total_train_step_per_epoch(self) -> int:
        # adjust to batch_size and world_size (sampler)
        return (
            self.nb_tokens // self.seq_length // self.batch_size // idr_torch.world_size
        )

    @property
    def total_train_step(self) -> int:
        return self.total_train_step_per_epoch * self.epochs

    def __str__(self) -> str:
        r"""
        Prints the configuration in a pretty multiline way. Each field will be
        printed on a different line to improve readability.
        """
        string = f"{self.__class__.__qualname__}(\n"
        for f in fields(self):
            string += " " * 4 + f"{f.name}={self.__getattribute__(f.name)},\n"
        string += ")"
        return string
