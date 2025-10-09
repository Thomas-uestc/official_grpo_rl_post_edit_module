# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val") -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val") -> None:
        prefix = "[TRAIN]" if context == "train" else "[VAL]"
        for inp, out, lab, score in samples:
            print(f"{prefix} [prompt] {inp}\n{prefix} [output] {out}\n{prefix} [ground_truth] {lab}\n{prefix} [score] {score}\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def __init__(self):
        # Initialize separate persistent tables for train and validation
        self.validation_table = None
        self.training_table = None
    
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val") -> None:
        # Determine table based on context
        if context == "train":
            table_attr = "training_table"
            log_key = "train/generations"
        else:
            table_attr = "validation_table"
            log_key = "val/generations"
        
        # Create column names for current samples (excluding input column)
        columns = ["step"] + sum(
            [[f"output_{i + 1}", f"label_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))],
            [],
        )
        
        # Get current table
        current_table = getattr(self, table_attr)
        
        # Create new table if first call
        if current_table is None:
            current_table = wandb.Table(columns=columns)
            setattr(self, table_attr, current_table)
        
        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=current_table.data)
        
        # Add new row with all data (excluding input from each sample)
        row_data = [step]
        for sample in samples:
            # sample format: (input, output, label, score)
            # We only want (output, label, score)
            _, output, label, score = sample
            row_data.extend([output, label, score])
        
        new_table.add_data(*row_data)
        
        # Log the updated table and update reference
        wandb.log({log_key: new_table}, step=step)
        setattr(self, table_attr, new_table)


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val") -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = "\n\n---\n\n".join(
                (f"input: {sample[0]}", f"output: {sample[1]}", f"label: {sample[2]}", f"score: {sample[3]}")
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


@dataclass
class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str]):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger]())

    def log(self, samples: List[Tuple[str, str, str, float]], step: int, context: str = "val") -> None:
        for logger in self.loggers:
            logger.log(samples, step, context)
