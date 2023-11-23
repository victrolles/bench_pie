from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, TypeVar

import idr_torch
import pandas as pd
import torch
from pandas import DataFrame, Series
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import BatchEncoding, PreTrainedTokenizer

from .config import TrainingConfig

T = TypeVar("T", bound="Ticket")


def _tokenize(text: str, tokenizer: PreTrainedTokenizer) -> list[int]:
    return tokenizer(text, truncation=False, add_special_tokens=False, verbose=False)[
        "input_ids"
    ]


class TicketDict(TypedDict):
    flags: str
    instruction: str
    category: str
    intent: str
    response: str


@dataclass
class Ticket:
    flags: str = field()
    instruction: str = field()
    category: str = field()
    intent: str = field()
    response: str = field()

    @classmethod
    def from_series(cls: type[T], series: Series) -> T:
        return cls(
            flags=series["flags"],
            instruction=series["instruction"],
            category=series["category"],
            intent=series["intent"],
            response=series["response"],
        )

    def format(
        self,
        instruction_prefix: str,
        response_prefix: str,
        category_prefix: str,
        intent_prefix: str
    ) -> str:
        # \n should be in prefix
        ticket_str = f"{instruction_prefix} {self.instruction}"
        ticket_str += f"{category_prefix} {self.category}"
        ticket_str += f"{intent_prefix} {self.intent}"
        ticket_str += f"{response_prefix} {self.response}"

        return ticket_str

    def tokenize(
        self,
        instruction_prefix: str,
        category_prefix: str,
        intent_prefix: str,
        response_prefix: str,
        tokenizer: PreTrainedTokenizer,
        target_only_preds: bool = True,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        input_encoding: list[int] = []
        target_encoding: list[int] = []

        instruction_prefix_encoding = _tokenize(instruction_prefix, tokenizer)
        category_prefix_encoding = _tokenize(category_prefix, tokenizer)
        intent_prefix_encoding = _tokenize(intent_prefix, tokenizer)
        response_prefix_encoding = _tokenize(response_prefix, tokenizer)

        instruction_encoding = _tokenize(self.instruction, tokenizer)
        category_encoding = _tokenize(self.category, tokenizer)
        intent_encoding = _tokenize(self.intent, tokenizer)
        response_encoding = _tokenize(self.response, tokenizer)

        input_encoding += instruction_prefix_encoding + instruction_encoding

        if target_only_preds:
            target_encoding += [pad_token_id] * len(input_encoding)

            input_encoding += category_prefix_encoding + category_encoding
            target_encoding += [pad_token_id] * len(category_prefix_encoding)
            target_encoding += category_encoding

            input_encoding += intent_prefix_encoding + intent_encoding
            target_encoding += [pad_token_id] * len(intent_prefix_encoding)
            target_encoding += intent_encoding

            input_encoding += response_prefix_encoding + response_encoding
            target_encoding += [pad_token_id] * len(response_prefix_encoding)
            target_encoding += response_encoding

        else:
            input_encoding += category_prefix_encoding + category_encoding
            input_encoding += intent_prefix_encoding + intent_encoding
            input_encoding += response_prefix_encoding + response_encoding

            target_encoding = input_encoding.copy()

        if tokenizer.bos_token_id is not None:
            input_encoding.insert(0, tokenizer.bos_token_id)
            target_encoding.insert(0, tokenizer.bos_token_id)

        input_encoding.append(tokenizer.eos_token_id)
        target_encoding.append(tokenizer.eos_token_id)

        input_tensor = torch.tensor(input_encoding, dtype=torch.int64)
        target_tensor = torch.tensor(target_encoding, dtype=torch.int64)
        return torch.stack([input_tensor, target_tensor], dim=-1)


class SupportDataset(ABC, torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        dataframe: DataFrame,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.dataframe = dataframe

        self.instruction_prefix = "\ninstruction: "
        self.category_prefix = "\ncategory: "
        self.intent_prefix = "\nintent: "
        self.response_prefix = "\nresponse: "

    def __len__(self) -> int:
        return len(self.dataframe)

    @abstractmethod
    def make_sampler(self) -> DistributedSampler | None:
        raise NotImplementedError()

    def make_dataloader(
        self, *, num_workers: int = 4, prefetch_factor: int = 2
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sampler=self.make_sampler(),
            collate_fn=self.collate,
        )

    def collate(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for the validation data loader.

        It will pad the input and target sequences to the maximum length of the batch.

        Args:
            batch (list): A list of tuples containing input ids and target ids.

        Returns:
            tuple: A tuple containing the input sequence tensor, input mask tensor, and
            target sequence tensor.
        """
        # First remove the last token of the input as well as the first token
        # of the target so that both sentences are shifted by 1 token.
        shifted_batch: list[torch.Tensor] = []
        for sample in batch:
            input_tensor, target_tensor = sample.unbind(dim=-1)
            shifted_input = input_tensor[:-1]
            shifted_target = target_tensor[1:]
            shifted_sample = torch.stack([shifted_input, shifted_target], dim=-1)
            shifted_batch.append(shifted_sample)

        # Now we want to do the padding. rnn.pad_sequence only pads on the right.
        # So we reverse the input, pad the batch and reverse the Tensor (padded batch).
        # The pad value is PAD_TOKEN_ID.
        padded = torch.nn.utils.rnn.pad_sequence(
            [sample.flip(dims=(0,)) for sample in shifted_batch],
            batch_first=True,
            padding_value=self.config.pad_token_id,
        ).flip(dims=(1,))
        input_tensor, target_tensor = padded.unbind(dim=-1)
        # for the training process, the mask should not be useful
        mask = torch.logical_not(
            (input_tensor == torch.full_like(input_tensor, self.config.pad_token_id))
        ).to(dtype=torch.int)
        return input_tensor, mask, target_tensor


class TrainSupportDataset(SupportDataset, torch.utils.data.IterableDataset):
    """Concatenate all tickets and slice them into sequence of fixed length.

    The dataset read json (IDRIS tickets) and html (IDRIS web pages) files.
    To have sample of equal length for our neural network, instead of padding tickets to
    have the same length, we prefer to concatenate or truncate tickets.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        dataframe: DataFrame,
    ):
        super().__init__(tokenizer, config, dataframe)
        self.rng = random.Random(53)

    def infinite_iterator(self) -> Iterator[Path]:
        while True:
            dataframe = self.dataframe.sample(frac=1)
            for _, row in dataframe.iterrows():
                yield row

    def get_next_sample(self, iterator: Iterator[Path]) -> torch.Tensor:
        """Get the next sample in the files list"""
        series = next(iterator)
        ticket = Ticket.from_series(series)

        return ticket.tokenize(
            instruction_prefix=self.instruction_prefix,
            category_prefix=self.category_prefix,
            intent_prefix=self.intent_prefix,
            response_prefix=self.response_prefix,
            tokenizer=self.tokenizer,
            target_only_preds=False,
            pad_token_id=self.config.pad_token_id,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        iterator = self.infinite_iterator()

        # buffer that will contain the token ids of the current CL_dataset sample
        all_token_ids: list[torch.Tensor] = []
        # buffer that will contain the token ids of the next ticket sample
        next_sample_ids = None
        # index of the current CL_dataset sample, needed for FSDP
        idx = 0

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        while True:
            if next_sample_ids is None:
                next_sample_ids = self.get_next_sample(iterator)

            if len(all_token_ids) + len(next_sample_ids) <= self.config.seq_length:
                # if the next HF_dataset sample can fit in the current CL_dataset sample
                # we add it
                all_token_ids += next_sample_ids
                next_sample_ids = None

            else:
                # if the next HF_dataset sample can't fit in the current CL_dataset
                # sample, we add what we can in the CL_dataset sample and then we yield
                # it
                # note: we add one more element compared to seq_length to return to
                # seq_length when generating inputs and targets (see train_collate())
                idx_break = self.config.seq_length - len(all_token_ids)
                all_token_ids += next_sample_ids[: idx_break + 1]
                next_sample_ids = next_sample_ids[idx_break + 1:]

                # yield the sample according to the rank of the dataloader worker
                # and the rank of the distributed training process
                if (
                    idx % (idr_torch.world_size * num_workers)
                    == idr_torch.rank * num_workers + worker_id
                ):
                    yield torch.stack(all_token_ids, dim=0)

                idx += 1
                all_token_ids = []

    def make_sampler(self) -> None:
        return None


class ValidSupportTicketDataset(SupportDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        dataframe: DataFrame,
    ):
        super().__init__(tokenizer, config, dataframe)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ticket = Ticket.from_series(self.dataframe.iloc[idx])
        return ticket.tokenize(
            instruction_prefix=self.instruction_prefix,
            category_prefix=self.category_prefix,
            intent_prefix=self.intent_prefix,
            response_prefix=self.response_prefix,
            tokenizer=self.tokenizer,
            target_only_preds=True,
            pad_token_id=self.config.pad_token_id,
        )

    def make_sampler(self) -> DistributedSampler:
        return DistributedSampler(
            self,
            num_replicas=idr_torch.world_size,
            rank=idr_torch.rank,
        )


def get_visu_data(
    dataset: ValidSupportTicketDataset, dataframe: DataFrame
) -> tuple[list[str], BatchEncoding, list[str]]:
    inp_prompt = []
    label_prompt = []

    for _, row in dataframe.iterrows():
        ticket = Ticket.from_series(row)

        inp_prompt.append(
            dataset.instruction_prefix
            + ticket.instruction
            + dataset.category_prefix
        )
        label_prompt.append(
            ticket.category
            + dataset.intent_prefix
            + ticket.intent
            + dataset.response_prefix
            + ticket.response
        )

    inp_tensor = dataset.tokenizer(inp_prompt, return_tensors="pt", padding=True)

    return inp_prompt, inp_tensor, label_prompt


def get_dataloaders(
    config: TrainingConfig, tokenizer: PreTrainedTokenizer
) -> tuple[DataLoader, DataLoader]:
    valid_dataframe = pd.read_csv(config.csv_data_path)
    train_dataframe = valid_dataframe.sample(frac=0.9, random_state=53)
    valid_dataframe = valid_dataframe.drop(train_dataframe.index)

    train_dataset = TrainSupportDataset(
        tokenizer, config=config, dataframe=train_dataframe
    )
    valid_dataset = ValidSupportTicketDataset(
        tokenizer, config=config, dataframe=valid_dataframe
    )

    visu_data = get_visu_data(
        valid_dataset, valid_dataframe.sample(n=2, random_state=53)
    )

    dataloader_kwargs = {
        "num_workers": 4,
        "prefetch_factor": 2,
    }
    train_loader = train_dataset.make_dataloader(**dataloader_kwargs)
    valid_loader = valid_dataset.make_dataloader(**dataloader_kwargs)
    return train_loader, valid_loader, visu_data
