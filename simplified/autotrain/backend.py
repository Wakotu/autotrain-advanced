import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Union

import requests
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from autotrain import logger
from autotrain.app_utils import run_training
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.dreambooth.params import DreamBoothTrainingParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.token_classification.params import TokenClassificationParams


_DOCKERFILE = """
FROM huggingface/autotrain-advanced:latest

CMD pip uninstall -y autotrain-advanced && pip install -U autotrain-advanced && autotrain api --port 7860 --host 0.0.0.0
"""

# format _DOCKERFILE
_DOCKERFILE = _DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()


@dataclass
class SpaceRunner:
    params: Union[
        TextClassificationParams,
        ImageClassificationParams,
        LLMTrainingParams,
        GenericParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        TokenClassificationParams,
    ]
    backend: str

    def __post_init__(self):
        self.spaces_backends = {
            "a10gl": "a10g-large",
            "a10gs": "a10g-small",
            "a100": "a100-large",
            "t4m": "t4-medium",
            "t4s": "t4-small",
            "cpu": "cpu-upgrade",
            "cpuf": "cpu-basic",
            "dgx-a100": "dgx-ngc",
            "nvcf-a100": "nvcf-ngc",
            "local": "local",
            "local-cli": "local-cli",
            "ep-aws-useast1-s": "aws_us-east-1_gpu_small_g4dn.xlarge",
            "ep-aws-useast1-m": "aws_us-east-1_gpu_medium_g5.2xlarge",
            "ep-aws-useast1-l": "aws_us-east-1_gpu_large_g4dn.12xlarge",
            "ep-aws-useast1-xl": "aws_us-east-1_gpu_xlarge_p4de",
            "ep-aws-useast1-2xl": "aws_us-east-1_gpu_2xlarge_p4de",
            "ep-aws-useast1-4xl": "aws_us-east-1_gpu_4xlarge_p4de",
            "ep-aws-useast1-8xl": "aws_us-east-1_gpu_8xlarge_p4de",
        }

        if not isinstance(self.params, GenericParams) and self.backend != "local-cli":
            if self.params.repo_id is not None:
                self.username = self.params.repo_id.split("/")[0]
            elif self.params.username is not None:
                self.username = self.params.username
            else:
                raise ValueError("Must provide either repo_id or username")
        else:
            self.username = self.params.username

        self.ep_api_url = (
            f"https://api.endpoints.huggingface.cloud/v2/endpoint/{self.username}"
        )

        if self.params.repo_id is None and self.params.username is not None:
            self.params.repo_id = f"{self.params.username}/{self.params.project_name}"

        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
        elif isinstance(self.params, TextClassificationParams):
            self.task_id = 2
        elif isinstance(self.params, TabularParams):
            self.task_id = 26
        elif isinstance(self.params, GenericParams):
            self.task_id = 27
        elif isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
        elif isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
        elif isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
        elif isinstance(self.params, TokenClassificationParams):
            self.task_id = 4
        else:
            raise NotImplementedError

    def prepare(self):
        if isinstance(self.params, LLMTrainingParams):
            self.task_id = 9
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TextClassificationParams):
            self.task_id = 2
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TabularParams):
            self.task_id = 26
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, GenericParams):
            self.task_id = 27
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, DreamBoothTrainingParams):
            self.task_id = 25
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, Seq2SeqParams):
            self.task_id = 28
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, ImageClassificationParams):
            self.task_id = 18
            space_id = self._create_space()
            return space_id
        if isinstance(self.params, TokenClassificationParams):
            self.task_id = 4
            space_id = self._create_space()
            return space_id
        raise NotImplementedError

    def _create_readme(self):
        _readme = "---\n"
        _readme += f"title: {self.params.project_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
        _readme += "---\n"
        _readme = io.BytesIO(_readme.encode())
        return _readme

    def _add_secrets(self, api, repo_id):
        if isinstance(self.params, GenericParams):
            for k, v in self.params.env.items():
                api.add_space_secret(repo_id=repo_id, key=k, value=v)
            self.params.env = {}

        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=self.params.token)
        api.add_space_secret(
            repo_id=repo_id, key="AUTOTRAIN_USERNAME", value=self.username
        )
        api.add_space_secret(
            repo_id=repo_id, key="PROJECT_NAME", value=self.params.project_name
        )
        api.add_space_secret(repo_id=repo_id, key="TASK_ID", value=str(self.task_id))
        api.add_space_secret(
            repo_id=repo_id, key="PARAMS", value=self.params.model_dump_json()
        )

        if isinstance(self.params, DreamBoothTrainingParams):
            api.add_space_secret(
                repo_id=repo_id, key="DATA_PATH", value=self.params.image_path
            )
        else:
            api.add_space_secret(
                repo_id=repo_id, key="DATA_PATH", value=self.params.data_path
            )

        if not isinstance(self.params, GenericParams):
            api.add_space_secret(repo_id=repo_id, key="MODEL", value=self.params.model)
            api.add_space_secret(
                repo_id=repo_id, key="OUTPUT_MODEL_REPO", value=self.params.repo_id
            )

    def _create_endpoint(self):
        hardware = self.spaces_backends[self.backend]
        accelerator = hardware.split("_")[2]
        instance_size = hardware.split("_")[3]
        region = hardware.split("_")[1]
        vendor = hardware.split("_")[0]
        instance_type = hardware.split("_")[4]
        payload = {
            "accountId": self.username,
            "compute": {
                "accelerator": accelerator,
                "instanceSize": instance_size,
                "instanceType": instance_type,
                "scaling": {"maxReplica": 1, "minReplica": 1},
            },
            "model": {
                "framework": "custom",
                "image": {
                    "custom": {
                        "env": {
                            "HF_TOKEN": self.params.token,
                            "AUTOTRAIN_USERNAME": self.username,
                            "PROJECT_NAME": self.params.project_name,
                            "PARAMS": self.params.model_dump_json(),
                            "DATA_PATH": self.params.data_path,
                            "TASK_ID": str(self.task_id),
                            "MODEL": self.params.model,
                            "OUTPUT_MODEL_REPO": self.params.repo_id,
                            "ENDPOINT_ID": f"{self.username}/{self.params.project_name}",
                        },
                        "health_route": "/",
                        "port": 7860,
                        "url": "public.ecr.aws/z4c3o6n6/autotrain-api:latest",
                    }
                },
                "repository": "autotrain-projects/autotrain-advanced",
                "revision": "main",
                "task": "custom",
            },
            "name": self.params.project_name,
            "provider": {"region": region, "vendor": vendor},
            "type": "protected",
        }
        headers = {"Authorization": f"Bearer {self.params.token}"}
        r = requests.post(self.ep_api_url, json=payload, headers=headers, timeout=120)
        return r.json()["name"]

    def _create_space(self):
        if (
            self.backend.startswith("dgx-")
            or self.backend.startswith("nvcf-")
            or self.backend.startswith("local")
        ):
            env_vars = {
                "HF_TOKEN": self.params.token,
                "AUTOTRAIN_USERNAME": self.username,
                "PROJECT_NAME": self.params.project_name,
                "TASK_ID": str(self.task_id),
                "PARAMS": json.dumps(self.params.model_dump_json()),
            }
            if isinstance(self.params, DreamBoothTrainingParams):
                env_vars["DATA_PATH"] = self.params.image_path
            else:
                env_vars["DATA_PATH"] = self.params.data_path

            if not isinstance(self.params, GenericParams):
                env_vars["MODEL"] = self.params.model
                env_vars["OUTPUT_MODEL_REPO"] = self.params.repo_id

            if self.backend.startswith("dgx-"):
                ngc_runner = NGCRunner(
                    job_name=self.params.repo_id.replace("/", "-"),
                    env_vars=env_vars,
                    backend=self.backend,
                )
                ngc_runner.create()
                return
            elif self.backend.startswith("nvcf-"):
                nvcf_runner = NVCFRunner(
                    job_name=self.params.repo_id.replace("/", "-"),
                    env_vars=env_vars,
                    backend=self.backend,
                )
                nvcf_runner.create()
                return
            else:
                local_runner = LocalRunner(
                    env_vars=env_vars, wait=self.backend == "local-cli"
                )
                pid = local_runner.create()
                return pid

        if self.backend.startswith("ep-"):
            endpoint_id = self._create_endpoint()
            return endpoint_id

        api = HfApi(token=self.params.token)
        repo_id = f"{self.username}/autotrain-{self.params.project_name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware=self.spaces_backends[self.backend.split("-")[1].lower()],
            private=True,
        )
        self._add_secrets(api, repo_id)
        readme = self._create_readme()
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

        _dockerfile = io.BytesIO(_DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=repo_id,
            repo_type="space",
        )
        return repo_id


@dataclass
class LocalRunner:
    env_vars: dict
    wait: bool = False

    def create(self):
        logger.info("Starting local training...")
        params = self.env_vars["PARAMS"]
        task_id = int(self.env_vars["TASK_ID"])
        training_pid = run_training(params, task_id, local=True, wait=self.wait)
        if not self.wait:
            logger.info(f"Training PID: {training_pid}")
        return training_pid
