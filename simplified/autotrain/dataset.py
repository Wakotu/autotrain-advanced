import io
import os
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from autotrain.preprocessor.text import (
    LLMPreprocessor,
)
from autotrain.preprocessor.vision import ImageClassificationPreprocessor


def remove_non_image_files(folder):
    # Define allowed image file extensions
    allowed_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]

            # If the file extension is not in the allowed list, remove the file
            if file_extension.lower() not in allowed_extensions:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

        # Recursively call the function on each subfolder
        for subfolder in dirs:
            remove_non_image_files(os.path.join(root, subfolder))


@dataclass
class AutoTrainDataset:
    train_data: List[str]
    task: str
    token: str
    project_name: str
    username: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    valid_data: Optional[List[str]] = None
    percent_valid: Optional[float] = None
    convert_to_class_label: Optional[bool] = False
    local: bool = False
    ext: Optional[str] = "csv"

    def __str__(self) -> str:
        info = f"Dataset: {self.project_name} ({self.task})\n"
        info += f"Train data: {self.train_data}\n"
        info += f"Valid data: {self.valid_data}\n"
        info += f"Column mapping: {self.column_mapping}\n"
        return info

    def __post_init__(self):
        if self.valid_data is None:
            self.valid_data = []
        if not self.valid_data and self.percent_valid is None:
            self.percent_valid = 0.2
        elif self.valid_data and self.percent_valid is not None:
            raise ValueError("You can only specify one of valid_data or percent_valid")
        elif self.valid_data:
            self.percent_valid = 0.0

        self.train_df, self.valid_df = self._preprocess_data()

    def _preprocess_data(self):
        train_df = []
        for file in self.train_data:
            if isinstance(file, pd.DataFrame):
                train_df.append(file)
            else:
                if self.ext == "jsonl":
                    train_df.append(pd.read_json(file, lines=True))
                else:
                    train_df.append(pd.read_csv(file))
        if len(train_df) > 1:
            train_df = pd.concat(train_df)
        else:
            train_df = train_df[0]

        valid_df = None
        if len(self.valid_data) > 0:
            valid_df = []
            for file in self.valid_data:
                if isinstance(file, pd.DataFrame):
                    valid_df.append(file)
                else:
                    if self.ext == "jsonl":
                        valid_df.append(pd.read_json(file, lines=True))
                    else:
                        valid_df.append(pd.read_csv(file))
            if len(valid_df) > 1:
                valid_df = pd.concat(valid_df)
            else:
                valid_df = valid_df[0]
        return train_df, valid_df

    @property
    def num_samples(self):
        return (
            len(self.train_df) + len(self.valid_df)
            if self.valid_df is not None
            else len(self.train_df)
        )

    def prepare(self):

        if self.task == "lm_training":
            text_column = self.column_mapping["text"]
            prompt_column = self.column_mapping.get("prompt")
            rejected_text_column = self.column_mapping.get("rejected_text")
            preprocessor = LLMPreprocessor(
                train_data=self.train_df,
                text_column=text_column,
                prompt_column=prompt_column,
                rejected_text_column=rejected_text_column,
                username=self.username,
                project_name=self.project_name,
                valid_data=self.valid_df,
                test_size=self.percent_valid,
                token=self.token,
                seed=42,
                local=self.local,
            )
            return preprocessor.prepare()

        else:
            raise ValueError(f"Task {self.task} not supported")
