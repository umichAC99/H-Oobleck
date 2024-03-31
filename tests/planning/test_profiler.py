import os
from pathlib import Path

import torch
import yaml
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    cleanup_temp_dir,
    initialize_temp_directories,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from oobleck.planning.profiler import ModelProfiler

from ..conftest import config, model_name, modules, tag
from .data_builder import GLUEDataBuilder

microbatch_size = 1

# def test_profile_model(tmp_path: Path, model: PreTrainedModel):
#     profiler = ModelProfiler(
#         tag=tag,
#         model_class_name=model_name,
#         config=config,
#         microbatch_size=microbatch_size,
#         base_dir=tmp_path,
#     )
#     assert not profiler.profile_exists()

#     batch = model.dummy_inputs
#     batch["labels"] = batch["input_ids"]
#     profiler.profile(batch)

#     assert profiler.profile_exists() is True
#     with open(profiler.profile_path) as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)
#     assert [row["layer_name"] for row in rows] == modules


class TestProfileModelClass(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        initialize_temp_directories()
        self._spawn_processes()

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @parametrize("precision", ["fp16", "bf16", "fp32"])
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_profile_model(self, precision: str):
        temp_path = Path(os.environ["TEMP_DIR"])
        profile_dir = temp_path / tag / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        torch.cuda.set_device(self.rank)

        dataloader = GLUEDataBuilder("gpt2").dataloader(batch_size=16)
        inputs = next(iter(dataloader))

        ModelProfiler._profile_model(
            model_name_or_path=model_name,
            model_config=config,
            optimizer_class="torch.optim.Adam",
            profile_dir=profile_dir,
            local_rank=self.rank,
            tp_size=self.world_size,
            precision=precision,
            inputs=inputs,
            warmup=1,
        )

        microbatch_size = inputs["input_ids"].shape[0]
        profile_path = (
            profile_dir
            / f"profile_tp{self.world_size}_mb{microbatch_size}_{precision}.yaml"
        )

        assert profile_path.exists()
        data = yaml.safe_load(profile_path.read_text())
        assert data["precision"] == precision
        assert data["tp_size"] == self.world_size
        assert data["model_name"] == model_name
        assert len(data["layers"]) == len(modules)
        assert [layer["layer_name"] for layer in data["layers"]] == modules


instantiate_parametrized_tests(TestProfileModelClass)
