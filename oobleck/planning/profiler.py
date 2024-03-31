import functools
import importlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import reduce
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.booster.plugin.hybrid_parallel_plugin import get_param_info
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


class EventTiming(Enum):
    FORWARD_START = 0
    FORWARD_END = 1
    BACKWARD_START = 2
    BACKWARD_END = 3
    OPTIMIZER_STEP_START = 4
    OPTIMIZER_STEP_END = 5


@dataclass
class ProfileData:
    module_name: str
    events: dict[EventTiming, torch.cuda.Event] = field(default_factory=dict)
    memory: dict[EventTiming, int] = field(default_factory=dict)


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
        profile_dir: Path,
        local_rank: int,
        tp_size: int,
        precision: str,
        inputs: dict[str, torch.Tensor],
        warmup: int = 3,
    ):
        store_path = profile_dir / "store"
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
        profile_data: dict[str, ProfileData] = {
            layer_name: ProfileData(
                module_name=layer_name,
                events={
                    timing: torch.cuda.Event(enable_timing=True)
                    for timing in EventTiming
                },
            )
            for layer_name in layers
        }

        optim_name, cls = optimizer_class.rsplit(".", 1)
        module = importlib.import_module(optim_name)
        optim_cls = getattr(module, cls)

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

        optimizer = optim_cls(model.parameters())
        optim_param_info = get_param_info(optimizer)
        if precision in ["fp16", "bf16"]:
            optimizer = MixedPrecisionOptimizer(optimizer, precision=precision)
        else:
            optimizer = OptimizerWrapper(optimizer)

        # Configure hooks for each layer
        def forward_pre_hook(module_name: str, module: nn.Module, inputs):
            module.to("cuda")
            profile_data[module_name].memory[EventTiming.FORWARD_START] = (
                torch.cuda.memory_allocated()
            )
            event = profile_data[module_name].events[EventTiming.FORWARD_START]
            event.record()

        def forward_hook(module_name: str, module: nn.Module, inputs, outputs):
            profile_data[module_name].memory[EventTiming.FORWARD_END] = (
                torch.cuda.memory_allocated()
            )
            event = profile_data[module_name].events[EventTiming.FORWARD_END]
            event.record()
            module.to("cpu")

        def backward_pre_hook(module_name, module: nn.Module, grad_output):
            module.to("cuda")
            profile_data[module_name].memory[EventTiming.BACKWARD_START] = (
                torch.cuda.memory_allocated()
            )
            event = profile_data[module_name].events[EventTiming.BACKWARD_START]
            event.record()

        def backward_hook(module_name, module: nn.Module, grad_input, grad_output):
            profile_data[module_name].memory[EventTiming.BACKWARD_END] = (
                torch.cuda.memory_allocated()
            )
            event = profile_data[module_name].events[EventTiming.BACKWARD_END]
            event.record()
            module.to("cpu")

        # Move inputs to cuda
        for name in inputs.keys():
            inputs[name] = inputs[name].to("cuda")

        logger.info("Profiler started...")

        for layer_name in layers:
            module = ModelProfiler.get_module_by_name(model, layer_name)

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
            logger.debug("Iterating until overflow solved...")
            outputs = model(**inputs)
            optimizer.backward(outputs.loss)
            torch.cuda.synchronize()

            if (
                mixed_precision is None
                or not optimizer.mixed_precision.should_skip_step()
            ):
                should_continue = False
                for param in model.parameters():
                    param.grad.data = param.grad.to("cpu")

                for layer_name in layers:
                    profile_data[layer_name].memory[
                        EventTiming.OPTIMIZER_STEP_START
                    ] = 0
                    profile_data[layer_name].memory[EventTiming.OPTIMIZER_STEP_END] = 0
                optimizer.step()

                num_parameters = 0
                working_to_master_map = optimizer.get_working_to_master_map()
                for layer_name in layers:
                    module = ModelProfiler.get_module_by_name(model, layer_name)
                    for p in module.parameters():
                        if precision in ["fp16", "bf16"]:
                            optim_param_index_id = optim_param_info["id2param"][
                                num_parameters
                            ]
                            master_tensor = working_to_master_map[optim_param_index_id]
                            states: dict[torch.Tensor, dict] = (
                                optimizer.optim.state.get(master_tensor)
                            )
                        else:
                            states: dict[torch.Tensor, dict] = (
                                optimizer.optim.state.get(p)
                            )

                        if states:
                            for name, state in states.items():
                                if isinstance(state, torch.Tensor):
                                    profile_data[layer_name].memory[
                                        EventTiming.OPTIMIZER_STEP_END
                                    ] += state.numel() * state.element_size()

                        num_parameters += 1

            optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

        logger.debug("Profiler finished.")

        if dist.get_rank() == 0:
            microbatch_size = inputs["input_ids"].shape[0]
            profile_path = (
                profile_dir
                / f"profile_tp{tp_size}_mb{microbatch_size}_{precision}.yaml"
            )
            logger.debug(f"Writing results to {profile_path}")

            data = {
                "model_name": model_name_or_path,
                "model_config": model_config.to_dict(),
                "microbatch_size": microbatch_size,
                "tp_size": tp_size,
                "precision": precision,
                "layers": [],
            }
            for index, (layer_name, layer_profile) in enumerate(profile_data.items()):
                data["layers"].append(
                    asdict(
                        LayerExecutionResult(
                            layer_index=index,
                            layer_name=layer_name,
                            forward=layer_profile.events[
                                EventTiming.FORWARD_START
                            ].elapsed_time(
                                layer_profile.events[EventTiming.FORWARD_END]
                            ),
                            backward=layer_profile.events[
                                EventTiming.BACKWARD_START
                            ].elapsed_time(
                                layer_profile.events[EventTiming.BACKWARD_END]
                            ),
                            mem_required=(
                                layer_profile.memory[EventTiming.FORWARD_END]
                                - layer_profile.memory[EventTiming.FORWARD_START]
                            )
                            + (
                                layer_profile.memory[EventTiming.BACKWARD_END]
                                - layer_profile.memory[EventTiming.BACKWARD_START]
                            )
                            + (
                                layer_profile.memory[EventTiming.OPTIMIZER_STEP_END]
                                - layer_profile.memory[EventTiming.OPTIMIZER_STEP_START]
                            ),
                        )
                    )
                )

            with profile_path.open("w") as f:
                yaml.safe_dump(data, f)

        dist.barrier()
        torch.cuda.synchronize()
