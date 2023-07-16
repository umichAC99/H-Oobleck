from __future__ import annotations

import functools
import logging
from enum import Enum

import torch
import torch.distributed
import torch.fx
from torch.distributed import ProcessGroup
from torch.distributed.fsdp._common_utils import HandleTrainingState
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy
from torch.utils.checkpoint import checkpoint as checkpoint_fn


class StreamType(Enum):
    UNSHARD = "unshard"
    UNSHARD_GRAD = "unshard_grad"
    EXECUTION = "execution"


class FullyShardedDataParallelLayer(torch.nn.Module):
    """
    Copy parameters in `layer` to CUDA device, then flatten and shard it.
    In GPU memory, there is only a `FlatParameter` object.

    After initialization, this layer always has sharded parameters.
    During `forward()`, it unshards the parameter, use it, and reshards it.
    For `backward()`, it breaks torch.autograd rule and manually accepts gradients
    and feed it into `torch.autograd.backward()` after unsharding.
    """

    def __init__(
        self,
        layer: torch.fx.GraphModule,
        process_group: ProcessGroup,
        streams: dict[StreamType, torch.cuda.Stream],
    ):
        super().__init__()
        device = torch.device("cuda", torch.cuda.current_device())
        layer.to(device)
        self._checkpointable = FullyShardedDataParallelLayer.is_checkpointable(layer)

        self._param_handle = FlatParamHandle(
            params=layer.parameters(),
            fully_sharded_module=layer,
            device=device,
            sharding_strategy=HandleShardingStrategy.FULL_SHARD,
            offload_params=False,
            mp_param_dtype=torch.float32,  # TODO: change to bf16
            mp_reduce_dtype=torch.float32,
            keep_low_precision_grads=False,
            process_group=process_group,
            use_orig_params=False,
        )
        self._param_handle.shard()
        self._param_handle._fully_sharded_module.register_full_backward_pre_hook(
            functools.partial(self._pre_backward_hook, self)
        )
        self._param_handle.flat_param.register_hook(
            functools.partial(self._post_backward_hook, self)
        )

        self._process_group = process_group
        self._streams = streams
        self._rank_index = torch.distributed.get_rank(group=process_group)
        self._group_size = torch.distributed.get_world_size(group=process_group)

        self.unshard_param_event = torch.cuda.Event()
        # TODO: register pre-backward hooks and post-backward hooks

    @staticmethod
    def is_checkpointable(layer: torch.fx.GraphModule) -> bool:
        if any(isinstance(m, torch.nn.Embedding) for _, m in layer.named_modules()):
            return False
        if any(
            isinstance(m, torch.nn.CrossEntropyLoss) for _, m in layer.named_modules()
        ):
            return False
        if next(layer.parameters(), None) is None:
            return False
        return True

    def unshard(self, state: HandleTrainingState):
        """
        Initiate unsharding of parameters (if not in progress).
        Mark all future execution to execute after unsharding to complete if `wait` is True.
        NOTE: it does not synchronize by blocking the current thread.
        """
        if self.unshard_param_event.query():
            # Unsharding either not started or already finished.
            if not self._param_handle.needs_unshard():
                # If already finished, return immediately
                return

            # If we don't have an event, it means there is no unsharding
            # in process. Iniiate unsharing

            assert state in [
                HandleTrainingState.FORWARD,
                HandleTrainingState.BACKWARD_PRE,
            ]
            self._param_handle._training_state = state

            unshard_stream = self._streams[StreamType.UNSHARD]
            execution_stream = self._streams[StreamType.EXECUTION]

            # unsharding must be done after execution
            unshard_stream.wait_stream(execution_stream)

            with torch.cuda.stream(unshard_stream):
                self._param_handle.pre_unshard()
                self._param_handle.unshard()
                self._param_handle.post_unshard()
                self.unshard_param_event.record(unshard_stream)

            # further execution should wait for unsharding to be done
            execution_stream.wait_stream(unshard_stream)

    def reshard(self):
        if self._param_handle.is_sharded(self._param_handle.flat_param):
            return

        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = self._streams[StreamType.EXECUTION]

        # resharding must be done after execution
        unshard_stream.wait_stream(execution_stream)

        with torch.cuda.stream(unshard_stream):
            self._param_handle.reshard(True)
            self._param_handle.post_reshard()

        # further execution should wait for unsharding to be done
        execution_stream.wait_stream(unshard_stream)
        self._param_handle._training_state = HandleTrainingState.IDLE

    def reshard_grad(self):
        # resharding must be done after execution
        unshard_stream = self._streams[StreamType.UNSHARD]
        execution_stream = self._streams[StreamType.EXECUTION]
        unshard_stream.wait_stream(execution_stream)

        with torch.cuda.stream(unshard_stream):
            self._param_handle.reshard_grad()

    def forward(self, *args) -> tuple[torch.Tensor]:
        self.unshard(HandleTrainingState.FORWARD)
        # TODO: uncomment it with fix in backward recomputation.
        # Currently recomputing in backward doesn't have unsharded params.
        # if self._checkpointable:
        #     result = checkpoint_fn(self._param_handle._fully_sharded_module, *args)
        # else:
        #     result = self._param_handle._fully_sharded_module(*args)
        result = self._param_handle._fully_sharded_module(*args)
        self.reshard()

        return result

    @staticmethod
    def _pre_backward_hook(
        self: FullyShardedDataParallelLayer,
        module: torch.nn.Module,
        grad_output: torch.nn.modules.module._grad_t,
    ):
        self.unshard(HandleTrainingState.BACKWARD_PRE)
        self._param_handle.prepare_gradient_for_backward()

    @staticmethod
    def _post_backward_hook(
        self: FullyShardedDataParallelLayer,
        grad_output: torch.Tensor,
    ):
        # follow fsdp._runtime_utils._post_backward_pass()
        # that stores _param_handle.flat_param._saved_grad_shard
        self._param_handle._training_state = HandleTrainingState.BACKWARD_POST
        self._param_handle.flat_param._post_backward_called = True
        # unsharded_grad = self._param_handle.flat_param.grad.data
        unsharded_grad = grad_output.data
        world_size = torch.distributed.get_world_size(self._process_group)
        chunks = list(unsharded_grad.chunk(world_size))
        new_sharded_grad = torch.empty_like(chunks[0])  # padded
        self._param_handle.flat_param._saved_grad_shard = new_sharded_grad

        self.reshard()

    def backward(self, tensor: torch.Tensor | tuple[tuple[torch.Tensor], torch.Tensor]):
        if isinstance(tensor, torch.Tensor):
            tensor.backward()
        else:
            output, gradients = tensor
            torch.autograd.backward(output, gradients)


class OobleckFullyShardedDataParallel:
    """A variant of Deepspeed Zero 3 or FSDP for Oobleck 3D parallelism.
    `TrainingState` based PyTorch FullyShardedDataParallel implementation is not suitable to
    integrate FSDP with pipeline parallelism, where multiple forwards and backwards are executed
    consecutively, not potentially interleaved.
    Instead, Oobleck only manages its state as [parameter/gradient]_[sharded/unsharded].

    Instead, Oobleck implementation provides a FSDPStage object, which is a list of layers.
    Each stage provides explicitly modularized functions that are executed in some order that works
    in conjunction with pipeline parallelism.
    """

    def __init__(self, layers: list[FullyShardedDataParallelLayer]):
        self._layers = layers
        self._streams: dict[str, torch.cuda.Stream] = {
            # stream for default computation
            "default": torch.cuda.current_stream(),
            # stream for unsharding parameters in forward pass
            "forward": torch.cuda.Stream(),
            # stream for unsharding gradients in backward pass
            "backward": torch.cuda.Stream(),
        }
