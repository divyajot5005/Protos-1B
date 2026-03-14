from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator

import torch
from datasets import IterableDataset, load_dataset


@dataclass
class DataSource:
    path: str
    name: str | None = None
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True
    shuffle_buffer: int = 10000


@dataclass
class DatasetConfig:
    sources: list[DataSource] = field(default_factory=list)
    sequence_length: int = 2048
    num_workers: int = 0
    seed: int = 7
    max_samples: int | None = None
    validation_sources: list[DataSource] = field(default_factory=list)
    test_mode: bool = False


class ResumablePackedDataset:
    def __init__(self, config: DatasetConfig, tokenizer, validation: bool = False) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.validation = validation
        self.sources = config.validation_sources if validation and config.validation_sources else config.sources
        self.sequence_length = config.sequence_length
        self.text_field = "text"
        self.state = {
            "source_index": 0,
            "records_consumed_in_source": 0,
            "buffer": [],
            "sequences_emitted": 0,
            "exhausted": False,
        }

    def state_dict(self) -> dict:
        return {
            "source_index": int(self.state["source_index"]),
            "records_consumed_in_source": int(self.state["records_consumed_in_source"]),
            "buffer": list(self.state["buffer"]),
            "sequences_emitted": int(self.state["sequences_emitted"]),
            "exhausted": bool(self.state["exhausted"]),
            "validation": self.validation,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.state["source_index"] = int(state.get("source_index", 0))
        self.state["records_consumed_in_source"] = int(state.get("records_consumed_in_source", 0))
        self.state["buffer"] = list(state.get("buffer", []))
        self.state["sequences_emitted"] = int(state.get("sequences_emitted", 0))
        self.state["exhausted"] = bool(state.get("exhausted", False))

    def _test_records(self) -> Iterator[dict[str, str]]:
        examples = [
            {"text": "Sparse update training can reduce optimizer overhead while keeping a dense forward pass."},
            {"text": "This is a short smoke test for the causal language model pretraining scaffold."},
            {"text": "Grouped query attention and rotary embeddings are enabled in this transformer implementation."},
        ]
        yield from itertools.cycle(examples)

    def _source_iter(self, source: DataSource) -> Iterator[dict[str, str]]:
        dataset = load_dataset(
            path=source.path,
            name=source.name,
            split=source.split,
            streaming=source.streaming,
        )
        if isinstance(dataset, IterableDataset):
            dataset = dataset.shuffle(seed=self.config.seed, buffer_size=source.shuffle_buffer)
        for record in dataset:
            text = record[source.text_field]
            if text:
                yield {"text": text}

    def _record_iter(self) -> Iterator[dict[str, str]]:
        if self.config.test_mode:
            yield from self._test_records()
            return

        if not self.sources:
            raise ValueError("At least one data source must be configured.")

        start_source_index = self.state["source_index"]
        skip_in_source = self.state["records_consumed_in_source"]
        for source_index in range(start_source_index, len(self.sources)):
            source_iter = self._source_iter(self.sources[source_index])
            to_skip = skip_in_source if source_index == start_source_index else 0
            for _ in range(to_skip):
                try:
                    next(source_iter)
                except StopIteration:
                    break
            self.state["records_consumed_in_source"] = to_skip
            for record in source_iter:
                self.state["source_index"] = source_index
                self.state["records_consumed_in_source"] += 1
                yield record
            self.state["source_index"] = source_index + 1
            self.state["records_consumed_in_source"] = 0

        self.state["exhausted"] = True

    def next_sequence(self) -> dict[str, torch.Tensor]:
        if self.state["exhausted"] and len(self.state["buffer"]) < self.sequence_length:
            raise StopIteration

        max_samples = self.config.max_samples
        if max_samples is not None and self.state["sequences_emitted"] >= max_samples:
            raise StopIteration

        buffer = list(self.state["buffer"])
        record_iter = self._record_iter()
        while len(buffer) < self.sequence_length:
            record = next(record_iter)
            token_ids = self.tokenizer(record[self.text_field], add_special_tokens=False)["input_ids"]
            buffer.extend(token_ids + [self.tokenizer.eos_token_id])

        chunk = buffer[: self.sequence_length]
        self.state["buffer"] = buffer[self.sequence_length :]
        self.state["sequences_emitted"] += 1
        ids = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    def next_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = [self.next_sequence() for _ in range(batch_size)]
        input_ids = torch.stack([item["input_ids"] for item in batch]).to(device, non_blocking=True)
        labels = torch.stack([item["labels"] for item in batch]).to(device, non_blocking=True)
        return {"input_ids": input_ids, "labels": labels}


def build_streaming_dataset(config: DatasetConfig, tokenizer, validation: bool = False) -> ResumablePackedDataset:
    return ResumablePackedDataset(config, tokenizer, validation=validation)
