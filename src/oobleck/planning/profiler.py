import os
import math
import time
import torch
import torch.distributed as dist

from ast import literal_eval
from pathlib import Path
from deepspeed.utils.logging import logger
from typing import List, Dict, Any, Tuple, Optional

from oobleck.module.model import OobleckModel
from oobleck.module.layer import Layer


PROFILE_CACHE = "/tmp/oobleck/profiles"
num_warmup = 2
num_iteration = 3


class LayerExecutionResult:
    def __init__(
        self,
        layer_index: int,
        forward: float,
        backward: float,
        allreduce_in_node: Dict[int, float],
        allreduce_cross_nodes: Dict[int, float],
        num_elements: int,
    ):
        self.index = layer_index
        self.forward = forward
        self.backward = backward
        self.allreduce_in_node = allreduce_in_node
        self.allreduce_cross_nodes = allreduce_cross_nodes
        self.num_elements = num_elements

    def __repr__(self) -> str:
        return (
            f"LayerExecutionResult(index={self.index}, "
            f"forward={self.forward}, backward={self.backward}, "
            f"allreduce_in_node={self.allreduce_in_node}, "
            f"allreduce_cross_nodes={self.allreduce_cross_nodes}, "
            f"num_elements={self.num_elements})"
        )


def get_profile_results(model: OobleckModel) -> List[LayerExecutionResult]:
    """Get the profiling results.

    Returns:
        List[LayerExecutionResult]: A list of execution results per layer.
    """

    def get_cache(cache_path: str):
        file = Path(cache_path)
        assert (
            file.is_file()
        ), f"Cache {cache_path} does not exist. Run profiler and cache the results."
        logger.debug("Loading cache %s", cache_path)
        with file.open(mode="r") as f:
            return literal_eval(f.read())

    directory = f"{PROFILE_CACHE}/{model.model_name}-{model.model_tag}"

    layer_execution_result = get_cache(f"{directory}/layers")
    allreduce_across_nodes = get_cache(f"{directory}/allreduce_across_nodes")
    allreduce_in_node = get_cache(f"{directory}/allreduce_in_node")

    results: List[LayerExecutionResult] = []
    for layer, execution, ar_in_node, ar_across_nodes in zip(
        model.model,
        layer_execution_result,
        allreduce_in_node,
        allreduce_across_nodes,
    ):
        results.append(
            LayerExecutionResult(
                layer.index,
                execution["forward"],
                execution["backward"],
                ar_in_node,
                ar_across_nodes,
                execution["num_elements"],
            )
        )

    return results


class Profiler:
    """Oobleck Profiler that profiles execution latency, allreduce latency in node,
    allreduce latency across node for each layer of the model.

    To support large model profiling, we offload parameters layer by layer.
    """

    def __init__(
        self,
        model: OobleckModel,
    ):
        self.model = model

    def profile_execution_layers(self) -> List[Dict[str, float]]:
        assert dist.is_initialized()

        results: List[List[int]] = [[0, 0, 0]] * len(self.model.model)
        if dist.get_rank() == 0:
            for i in range(num_warmup + num_iteration):
                logger.info(f"Profiling layer execution ltency: {i} iteration")
                input = tuple(self.model.sample_inputs.values())

                for idx, layer in enumerate(self.model.model):
                    if isinstance(input, tuple):
                        input = tuple(
                            [
                                t.detach().to("cuda")
                                if isinstance(t, torch.Tensor)
                                else t
                                for t in input
                            ]
                        )
                    else:
                        input = input.detach().to("cuda")

                    gpu_layer = layer.to("cuda")
                    torch.cuda.synchronize()

                    start = time.time()
                    output = gpu_layer(*input)
                    torch.cuda.synchronize()
                    end = time.time()
                    input = output
                    forward = end - start

                    del gpu_layer
                    if i < num_warmup:
                        continue

                    results[idx][0] += forward * 1000
                    results[idx][1] += forward * 2000
                    if results[idx][2] == 0:
                        results[idx][2] = sum([p.numel() for p in layer.parameters()])

        for result in results:
            result[0] /= num_iteration
            result[1] /= num_iteration

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {"forward": result[0], "backward": result[1], "num_elements": result[2]}
            for result in results.tolist()
        ]

    @staticmethod
    def profile_allreduce_layer(
        layer: Layer, process_group: dist.ProcessGroup
    ) -> float:
        numel = sum([p.numel() for p in layer.parameters()])
        tensor = torch.zeros(numel, dtype=torch.float32, device="cuda")

        dist.barrier(process_group)
        start = time.time()
        dist.all_reduce(tensor, group=process_group)
        dist.barrier(process_group)
        end = time.time()

        del tensor
        return (end - start) * 1000

    def profile_allreduce_across_nodes(self) -> List[Dict[int, float]]:
        """Profile allreduce latency across nodes,
        \# nodes = 2, 3, ... N.
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            List[Dict[int, float]]: A list of allreduce latency,
            where key is the number of nodes and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(
            f"Profile allreduce acorss {os.environ['WORLD_SIZE']} nodes latency"
        )

        num_gpus_per_node = torch.cuda.device_count()
        ranks = list(range(0, dist.get_world_size(), num_gpus_per_node))

        process_groups: List[Tuple(bool, dist.ProcessGroup)] = []
        for i in range(1, len(ranks) + 1):
            pg_ranks = ranks[:i]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: List[List[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.model))
        ]
        for layer_index, layer in enumerate(self.model.model):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {len(ranks[:i]) + 1: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]

    def profile_allreduce_in_node(self) -> List[Dict[int, float]]:
        """Profile allreduce latency between GPUs in node,
        \# nodes = 1, 2, 4, ....
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            List[Dict[int, float]]: A list of allreduce latency,
            where key is the number of GPUs and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(f"Profile allreduce within a node latency")

        num_gpus_per_node = torch.cuda.device_count()
        # 1, 2, 4, 8, ...
        num_gpus_list = [2**i for i in range(int(math.log2(num_gpus_per_node)) + 1)]
        ranks = list(range(num_gpus_per_node))

        process_groups: List[Tuple(bool, dist.ProcessGroup)] = []
        for i in range(len(num_gpus_list)):
            pg_ranks = ranks[: num_gpus_list[i]]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: List[List[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.model))
        ]
        for layer_index, layer in enumerate(self.model.model):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()

        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {num_gpus_list[i]: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]


def profile(
    model_name: str,
    sample_inputs: Dict[str, Any],
    master_addr: str,
    master_port: int,
    world_size: int,
    rank: int,
    local_rank: int,
    model_tag: Optional[str] = None,
    model_args: Optional[Dict[str, Any]] = None,
):
    """Profile the given model and return a list of execution result
    per layer.
    ExecutionResult includes forward/backward latency, allreduce in node,
    and allreduce across nodes.

    Result is stored in cache for future use.
    Path: /tmp/oobleck/profiles/{model_name}-{tag}/{layers|allreduce_in_node|allreduce_across_nodes}
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    directory = f"{PROFILE_CACHE}/{model_name}-{model_tag}"
    if Path(directory).exists():
        logger.info("Profile results already exist. Skipping profiling.")
        return

    # assert dist.is_initialized(), "Distributed is not initialized."
    assert not dist.is_initialized(), "Distributed is already initialized."
    dist.init_process_group(backend="nccl")

    os.makedirs(directory, exist_ok=True)
    logger.info("Profiling model %s", model_name)

    model = OobleckModel(model_name, sample_inputs, None, model_tag, model_args)
    profiler = Profiler(model)
    # forward/backward execution
    layer_execution_result = profiler.profile_execution_layers()
    allreduce_across_nodes = profiler.profile_allreduce_across_nodes()
    allreduce_in_node = profiler.profile_allreduce_in_node()

    if "0" in os.environ["CUDA_VISIBLE_DEVICES"]:
        with Path(f"{directory}/layers").open(mode="w") as f:
            f.write(str(layer_execution_result))
            f.flush()
        with Path(f"{directory}/allreduce_across_nodes").open(mode="w") as f:
            f.write(str(allreduce_across_nodes))
            f.flush()
        with Path(f"{directory}/allreduce_in_node").open(mode="w") as f:
            f.write(str(allreduce_in_node))
            f.flush()
        logger.info("Profile data is stored in %s", directory)

    dist.barrier()
    dist.destroy_process_group()
