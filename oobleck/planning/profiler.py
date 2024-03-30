import csv
import functools
import importlib
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.interface import OptimizerWrapper
from colossalai.shardformer import ShardConfig, ShardFormer
from loguru import logger
from oobleck_colossalai.pipeline_template import PipelineTemplate
from torch.distributed import FileStore
from transformers import PretrainedConfig, PreTrainedModel

from oobleck.engine.configuration_engine import ConfigurationEngine


@dataclass
class LayerExecutionResult:
    layer_index: int
    layer_name: str
    forward: float
    backward: float
    mem_required: int


class ModelProfiler:
    """A class for profiling a model.

    Profiling includes:
    - Forward and backward latency (in ms) for each layer
    - Maximum memory consumption (in bytes) for each layer

    Args:
        model (nn.Module): The model to be profiled.
        layers (list[str]): A list of layer names to be profiled.
        model must have modules with the given names.
    """

    def __init__(
        self,
        tag: str,
        model_name_or_path: str,
        optimizer_class: str,
        config: PretrainedConfig,
        precision: str,
        microbatch_size: int,
        tp_size: int,
        base_dir: Path,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_config = config
        self.microbatch_size = microbatch_size
        self.tp_size = tp_size
        self.profile_path = base_dir / tag / "profile" / f"mb_{microbatch_size}.csv"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def get_profile(
        data, inputs: dict[str, torch.Tensor]
    ) -> list[LayerExecutionResult]:
        # For agent_index == 0, run _load_profile() or _profile_model()
        # And then, rank==0 broadcasts the result
        # For others, receive data from rank==0
        # rank 0 must always broadcast data, since others may not have profile cache
        assert dist.is_initialized(), "torch.distributed is not initialized."

    def _load_profile(self):
        """If cache file exists, load it."""
        pass

    def _init_profile(self, inputs: dict[str, torch.Tensor]):
        """Profile the model with a new child process.

        All worker in the first agent calls this function.
        torch.distributed is initialized using FileStore.
        """
        configuration_engine = ConfigurationEngine.get_instance()
        assert configuration_engine.agent_index == 0, "Only agent 0 can profile."

        context = torch.multiprocessing.get_context("spawn")
        process = context.Process(
            target=ModelProfiler._profile_model,
            args=(
                self.model_name_or_path,
                self.model_config,
                self.profile_path,
                configuration_engine.local_rank,
                self.tp_size,
                inputs,
            ),
            daemon=True,
        )
        process.start()
        process.join()

    @staticmethod
    def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
        """Get a module by its name."""
        names = name.split(".")
        return reduce(getattr, names, model)

    @staticmethod
    def _profile_model(
        model_name_or_path: str,
        model_config: PretrainedConfig,
        optimizer_class: str,
        profile_path: Path,
        local_rank: int,
        tp_size: int,
        precision: str,
        inputs: dict[str, torch.Tensor],
        warmup: int = 3,
    ):
        store_path = profile_path.parent / "store"
        logger.debug(
            f"Profiler initiating torch.distributed: {store_path} with {tp_size} workers"
        )

        store = FileStore(str(store_path), tp_size)
        dist.init_process_group(
            backend="nccl",
            world_size=tp_size,
            rank=local_rank,
            store=store,
        )

        assert dist.get_world_size() == tp_size, "World size mismatch"
        logger.debug(f"Sharding model with {tp_size} ranks")

        module_name, cls = model_name_or_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model: PreTrainedModel = getattr(module, cls)(model_config).to("cpu")
        model.gradient_checkpointing_enable()
        layers = PipelineTemplate.get_modules(model)

        optim_name, cls = optimizer_class.rsplit(".", 1)
        module = importlib.import_module(optim_name)
        optim_cls = getattr(module, cls)

        optimizer = optim_cls(model.parameters())

        if tp_size > 1:
            shard_config = ShardConfig(
                tensor_parallel_process_group=dist.new_group(),
                pipeline_stage_manager=None,
                enable_tensor_parallelism=True,
                enable_flash_attention=True,
            )
            shardformer = ShardFormer(shard_config)

            model, _ = shardformer.optimize(model)

        mixed_precision = None
        if precision == "fp16":
            mixed_precision = torch.float16
        elif precision == "bf16":
            mixed_precision = torch.bfloat16
        if mixed_precision is not None:
            model = model.to(dtype=mixed_precision)

        if precision in ["fp16", "bf16"]:
            optimizer = MixedPrecisionOptimizer(optimizer, precision=precision)
        else:
            optimizer = OptimizerWrapper(optimizer)

        # Configure hooks for each layer
        def forward_pre_hook(module_name: str, module: nn.Module, inputs):
            module.to("cuda")
            memory[module][0] = torch.cuda.memory_allocated()
            events[module][0].record()

        def forward_hook(module_name: str, module: nn.Module, inputs, outputs):
            memory[module][1] = torch.cuda.memory_allocated()
            events[module][1].record()
            module.to("cpu")

        def backward_pre_hook(module_name, module: nn.Module, grad_output):
            module.to("cuda")
            memory[module][2] = torch.cuda.memory_allocated()
            events[module][2].record()

        def backward_hook(module_name, module: nn.Module, grad_input, grad_output):
            memory[module][3] = torch.cuda.memory_allocated()
            events[module][3].record()
            module.to("cpu")

        # Move inputs to cuda
        for name in inputs.keys():
            inputs[name] = inputs[name].to("cuda")
            inputs[name].requires_grad = inputs[name].is_floating_point()

        logger.info("Profiler started...")

        events: dict[nn.Module, list[torch.cuda.Event]] = {}
        memory: dict[nn.Module, list[int]] = {}

        for layer_name in layers:
            module = ModelProfiler.get_module_by_name(model, layer_name)

            # forward start, forward end, backward start, backward end
            events[module] = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            memory[module] = [0 for _ in range(4)]

            module.register_forward_pre_hook(
                functools.partial(forward_pre_hook, layer_name)
            )
            module.register_forward_hook(functools.partial(forward_hook, layer_name))
            module.register_full_backward_pre_hook(
                functools.partial(backward_pre_hook, layer_name)
            )
            module.register_full_backward_hook(
                functools.partial(backward_hook, layer_name)
            )

        with torch.no_grad():
            for _ in range(warmup):
                model(**inputs)

        should_continue: bool = True

        while should_continue:
            outputs = model(**inputs)
            loss = outputs.loss
            optimizer.backward(loss)
            if (
                mixed_precision is None
                or not optimizer.mixed_precision.should_skip_step()
            ):
                should_continue = False
                optimizer.step()

        torch.cuda.synchronize()

        logger.debug("Profiler finished.")

        if dist.get_rank() == 0:
            logger.debug(f"Writing results to {profile_path}")
            with profile_path.open("w") as f:
                writer = csv.DictWriter(
                    f, fieldnames=LayerExecutionResult.__annotations__.keys()
                )
                writer.writeheader()

                for index, layer_name in enumerate(layers):
                    module = ModelProfiler.get_module_by_name(model, layer_name)
                    forward = events[module][0].elapsed_time(events[module][1])
                    backward = events[module][2].elapsed_time(events[module][3])
                    mem_required = (memory[module][3] - memory[module][2]) + (
                        memory[module][1] - memory[module][0]
                    )

                    result = LayerExecutionResult(
                        layer_index=index,
                        layer_name=layer_name,
                        forward=forward,
                        backward=backward,
                        mem_required=mem_required,
                    )
                    writer.writerow(asdict(result))

            store_path.unlink()

            config_path = profile_path.parent / "config.yaml"
            with config_path.open("w") as f:
                yaml.safe_dump(model_config.to_dict(), f)

            # TODO: merge config, and profile data altogether.
            # Add microbatch_size and tp_size into file name

        dist.barrier()
        dist.destroy_process_group()
