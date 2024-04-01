import sys
from typing import Counter
from unittest.mock import patch

import torch
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from colossalai.interface import ModelWrapper, OptimizerWrapper
from oobleck_colossalai import (
    HeterogeneousDataLoader,
    HeterogeneousParallelModule,
    PipelineTemplate,
)
from torch.optim import Adam
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.execution_engine import ExecutionEngine
from oobleck.engine.plugin import OobleckPlugin

from ..conftest import config, init_profile_data, tag
from .conftest import (
    OobleckMultiprocessTestBase,
    template_1stage,
    template_2stages,
    template_3stages,
)
from .data_builder import GLUEDataBuilder


class OobleckEngineTestBase(OobleckMultiprocessTestBase):
    def prepare(
        self,
    ) -> tuple[ExecutionEngine, ModelWrapper, OptimizerWrapper, DataLoader]:
        self.init_oobleck()

        configuration_engine = ConfigurationEngine.get_instance()
        init_profile_data(
            configuration_engine.base_dir / tag / "profile",
            self.tp_size,
            self.microbatch_size,
            "fp32",
        )

        templates = [template_1stage, template_2stages, template_3stages]
        plugin = OobleckPlugin(
            tp_size=self.tp_size,
            global_batch_size=self.global_batch_size,
            microbatch_size=self.microbatch_size,
            precision="fp32",
        )

        engine = ExecutionEngine(plugin)

        dataloader = GLUEDataBuilder("gpt2", plugin).train_dataloader()
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        with (
            patch(
                "oobleck.engine.execution_engine.create_pipeline_templates",
                return_value=templates,
            ),
            patch.object(
                configuration_engine, "init_distributed", new=self.init_distributed
            ),
        ):
            model, optimizer, _, dataloader, lr_scheduler = engine.prepare(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                lr_scheduler=lr_scheduler,
            )

        return engine, model, optimizer, dataloader

    def do_step(
        self,
        engine: ExecutionEngine,
        model: ModelWrapper,
        optimizer: OptimizerWrapper,
        dataloader: DataLoader,
    ):
        engine.execute(
            iter(dataloader),
            model,
            lambda outputs, inputs: outputs.loss,
            optimizer,
            return_loss=True,
            return_outputs=True,
        )

        optimizer.step()
        optimizer.zero_grad()


class TestEngineExecutionClass(OobleckEngineTestBase):
    @parametrize(
        "num_hosts, tp_size",
        [(1, 4), (2, 2), (3, 1), (4, 1)],
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_engine_execute(self, num_hosts: int, tp_size: int):
        self.num_hosts = num_hosts
        self.tp_size = tp_size

        if self.rank >= self.world_size:
            sys.exit(0)

        engine, model, optimizer, dataloader = self.prepare()
        with (
            patch.object(
                model, "sync_dp_grads", wraps=model.sync_dp_grads
            ) as sync_mock,
            patch.object(
                engine.plugin.schedule,
                "forward_step",
                wraps=engine.plugin.schedule.forward_step,
            ) as forward_mock,
            patch.object(
                engine.plugin.schedule,
                "backward_step",
                wraps=engine.plugin.schedule.backward_step,
            ) as backward_mock,
        ):
            self.do_step(engine, model, optimizer, dataloader)

        dist.barrier()
        torch.cuda.synchronize()

        assert sync_mock.call_count == 1

        num_microbatches = engine.plugin.num_microbatches[
            engine.plugin.pipelines[engine.plugin._pipeline_index]
        ]
        assert forward_mock.call_count == num_microbatches
        assert backward_mock.call_count == num_microbatches


class TestEnginePrepareClass(OobleckEngineTestBase):
    backend: str = "gloo"
    num_hosts: int = 9
    tp_size: int = 2

    @parametrize(
        "pipelines",
        [
            [template_3stages, template_3stages, template_3stages],
            [template_3stages, template_2stages, template_2stages, template_2stages],
        ],
        name_fn=lambda pipelines: "homogeneous"
        if len(pipelines) == 3
        else "heterogeneous",
    )
    def test_engine_prepare(self, pipelines: list[PipelineTemplate]):
        with patch(
            "oobleck.engine.execution_engine.PipelineInstantiator.instantiate",
            return_value=(
                dict(Counter(pipelines)),
                {
                    template: self.global_batch_size // len(pipelines)
                    for template in pipelines
                },
            ),
        ):
            engine, model, optimizer, dataloader = self.prepare()

        assert isinstance(model, HeterogeneousParallelModule)
        assert isinstance(
            optimizer, (HybridParallelAMPOptimizer, HybridParallelNaiveOptimizer)
        )
        assert isinstance(dataloader, HeterogeneousDataLoader)
        assert (
            dataloader.batch_sampler and dataloader._DataLoader__initialized
        ), "HeterogeneousDataLoader.configure() is not called."

        assert dist.is_initialized()

        assert engine.plugin.pipelines == pipelines
        assert (
            sum(engine.plugin.num_microbatches[pipeline] for pipeline in pipelines)
            == self.global_batch_size
        )
        assert engine.plugin.dp_size == len(pipelines)


instantiate_parametrized_tests(TestEngineExecutionClass)
instantiate_parametrized_tests(TestEnginePrepareClass)
