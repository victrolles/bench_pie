#! /usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import json
from argparse import SUPPRESS, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from .config import TrainingConfig
from .train_model import train


def get_cfg_file_args(config_file: Path | None) -> dict[str, Any]:
    if config_file is None:
        return dict()

    with config_file.open("r") as cfg_file:
        match config_file.suffix:
            case ".json":
                defaults = json.load(cfg_file)
            case ".yaml":
                defaults = yaml.safe_load(cfg_file)
            case ".ini" | ".cfg":
                cfg = configparser.ConfigParser()
                cfg.read_file(cfg_file)
                defaults = {s: dict(cfg.items(s)) for s in cfg.sections()}
            case _:
                raise NotImplementedError("This file type is not acceptable.")
    return defaults


class BaseNamespaceProtocol(Protocol):
    action: str
    config_file: Path


class BaseNamespace(Namespace, BaseNamespaceProtocol):
    pass


def cli():
    parser = ArgumentParser("PIE")

    subparsers = parser.add_subparsers(dest="action")

    config_file_parser = ArgumentParser(add_help=False)
    config_file_parser.add_argument(
        "-c",
        "--config",
        "--configfile",
        "--config_file",
        "--config-file",
        dest="config_file",
        type=Path,
        default=None,
    )

    training_parser = subparsers.add_parser(
        "train",
        argument_default=SUPPRESS,
        parents=[config_file_parser],
    )
    TrainingConfig.add_args(training_parser)

    cli_args = cast(BaseNamespace, parser.parse_args())
    cfg_file_args = get_cfg_file_args(cli_args.config_file)
    if cli_args.action == "train":
        config = TrainingConfig.from_mappings(cfg_file_args, cli_args)
        train(config)
