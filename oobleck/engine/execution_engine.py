import itertools
import math
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.shardformer.policies.auto_policy import _fullname
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.pipeline_instantiator import PipelineInstantiator
from oobleck.engine.plugin import OobleckPlugin
from oobleck.planning.planner import create_pipeline_templates
from oobleck.planning.profiler import ModelProfiler


class ExecutionEngine:
    """A main execution engine using an execution Backend.

    ExecutionEngine does not have a global view of distributed training.

    """

    def __init__(
        self,
        plugin: OobleckPlugin,
        **booster_kwargs,
    ):
        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."
        assert isinstance(
            plugin, OobleckPlugin
        ), "Plugin must be an instance of OobleckPlugin."

        self.plugin = plugin

        self.pipeline_templates: list[PipelineTemplate] | None = None
        self.booster: Booster | None = None
        self.booster_kwargs = booster_kwargs

    @property
    def is_master(self) -> bool:
        configuration_engine = ConfigurationEngine.get_instance()
        if configuration_engine is None:
            raise RuntimeError("ConfigurationEngine must be initialized.")
        return configuration_engine.is_master

    def prepare(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        criterion: Callable | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> tuple[nn.Module, Optimizer, Callable, DataLoader, LRScheduler]:
        """Initialize pipeline templates and distributed configuration.

        This function automatically initializes torch.distributed,
        create pipeline templates, instantiate pipelines, and then boost the model.
        """

        assert (
            not dist.is_initialized()
        ), "Distributed environment must not be initialized."

        if self.pipeline_templates is not None:
            raise RuntimeError(
                "Pipeline templates are already initialized. "
                "You should not call prepare() more than once."
            )

        configuration_engine = ConfigurationEngine.get_instance()

        profiler = ModelProfiler(
            configuration_engine.tag,
            model_name_or_path=_fullname(model),
            optimize_class=_fullname(optimizer),
            model_config=model.config,
            precision=self.plugin.precision,
            tp_size=self.plugin.tp_size,
            base_dir=configuration_engine.base_dir,
        )

        profile_dataloder = DataLoader(
            dataloader.dataset, batch_size=self.plugin.microbatch_size
        )
        inputs = next(iter(profile_dataloder))
        profiler.init_profile(inputs)

        configuration_engine.init_distributed()
        profile_data = profiler.load_profile(self.plugin.microbatch_size)

        # Calculate the minimum number of nodes required
        memory = torch.cuda.get_device_properties(0).total_memory
        min_num_nodes = max(
            1,
            math.ceil(sum(layer.mem_required for layer in profile_data) / memory),
        )
        max_num_nodes = configuration_engine.world_size // self.plugin.tp_size

        logger.debug("Creating pipeline templates...")
        self.pipeline_templates = create_pipeline_templates(
            _fullname(model),
            self.plugin.microbatch_size,
            list(range(min_num_nodes, max_num_nodes + 1)),
            configuration_engine.base_dir / configuration_engine.tag / "profile",
        )

        logger.debug(f"Pipeline templates: {self.pipeline_templates}")

        pipeline_instantiator = PipelineInstantiator(
            self.pipeline_templates,
            self.plugin.global_batch_size // self.plugin.microbatch_size,
        )
        num_instances, num_microbatches = pipeline_instantiator.instantiate(
            len(configuration_engine.dist_info)
        )
        logger.debug(f"Pipeline instances: {num_instances}")
        logger.debug(f"Microbatches: {num_microbatches}")
        self.plugin.set_pipelines(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(template, num_templates)
                    for template, num_templates in num_instances.items()
                )
            ),
            num_microbatches,
        )
        self.booster = Booster(plugin=self.plugin, **self.booster_kwargs)
        return self.booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

    def _estimate_max_num_nodes_required(self):
        # TODO: implement it
        pass

    def execute(
        self,
        dataloader_iterator: Iterator,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict[str, Any]:
        return self.booster.execute_pipeline(
            dataloader_iterator,
            model,
            criterion,
            optimizer,
            return_loss=return_loss,
            return_outputs=return_outputs,
        )

    # TODO (insujang): Implement the following
    # load_model, save_model, load_optimizer, save_optimizer, load_lr_scheduler, save_lr_scheduler
