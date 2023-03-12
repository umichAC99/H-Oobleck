import math

from typing import List, Tuple
from deepspeed.utils.logging import logger

from oobleck.module.model import OobleckModel


class StageExecutionSpec:
    def __init__(self, layer_start_index: int, layer_end_index: int, ranks: List[int]):
        self.layer_start_index = layer_start_index
        self.layer_end_index = layer_end_index
        self.ranks = ranks

    def get_layer_indices(self) -> Tuple[int, int]:
        return self.layer_start_index, self.layer_end_index


class PipelineSpec:
    """
    A specification of the pipeline representation.
    Oobleck represents the total device cluster as a linear combination
    of several distinct heterogeneous PipelineSpecs to cover all availale GPUs.

    Based on the given fault tolerance spec and maximum available number of nodes,
    several PipelineSpecs are created in advance and the planner analyzes the optimal
    execution plan for each PipelineSpec under the given number of GPUs.

    Exploiting the Frobenius problem, it is guaranteed to represent any feasible number of nodes
    as a linear of combination of PipelineSpecs with consecutive number of nodes.
    """

    def __init__(self, num_nodes: int, num_gpus_per_node: int):
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node

    # TODO: move to planner part
    def create_optimal_plan(
        self, target_model: OobleckModel
    ) -> List[StageExecutionSpec]:
        """Create an optimal execution plan with the given number of GPUs
        using profiled execution information.

        Current Alpha-level implementation: divide layers individually.
        Number of stages is equal to the number of nodes.
        """
        num_layer_per_node = len(target_model.model) // self.num_nodes
        # num_layers: number of layers that are assigned to each stage.
        num_layers = [num_layer_per_node] * self.num_nodes
        if num_layer_per_node * self.num_nodes < len(target_model.model):
            num_layers[-1] += (
                len(target_model.model) - num_layer_per_node * self.num_nodes
            )

        stage_specs = []
        sum = 0
        for i, (node_id, num_layer) in enumerate(
            zip(range(self.num_nodes), num_layers)
        ):
            end_index = sum + num_layer
            stage_specs.append(
                StageExecutionSpec(
                    sum,
                    end_index,
                    list(range(i * node_id, i * node_id + self.num_gpus_per_node)),
                )
            )

            sum += num_layer

        return stage_specs


class PipelineSpecs:
    """
    A wrapper class of the list of :class:`.PipelineSpec`.
    It generates the list of :class:`.PipelineSpec`s that can represent any number N that
    min_num_nodes <= N <= max_num_nodes as a linear combination of them.
    """

    def __init__(
        self,
        ft_spec: int,
        min_num_nodes: int,
        max_num_nodes: int,
        num_gpus_per_node: int,
    ):
        self.pipeline_specs = None

        if max_num_nodes < min_num_nodes:
            # It is not possible to run model training if the model is too large to fit
            # in the GPU memory of the given nodes.
            return

        min_req_nodes = min_num_nodes * (ft_spec + 1)
        if max_num_nodes < min_req_nodes:
            logger.warning(
                "The number of nodes is not enough to provide at least ft_spec + 1 copy of the model."
                "Oobleck may fail to provide fault tolerancy if continue."
            )

        # Oobleck's requirements to solve the Frobenius problem
        # 1. p > n[0] - 2 (thus the minimum p is n[0] - 1)
        # 2. n's are contiguous integers (n[i] + 1 = n[i+1])
        # TODO: verify that it is always better to have smaller number of GPUs per PipelineSpecs.
        # (i.e. why we choose minimum p)
        num_pipeline_specs = min_num_nodes - 1
        if num_pipeline_specs < 1:
            num_pipeline_specs = 1

        pipeline_spec_num_nodes = list(
            range(min_num_nodes, min_num_nodes + num_pipeline_specs)
        )
        if any(num_nodes > max_num_nodes for num_nodes in pipeline_spec_num_nodes):
            # Some PipelineSpec cannot be realized with current max_num_nodes requirement.
            return

        self.ft_spec = ft_spec
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.pipeline_specs = [
            PipelineSpec(num_nodes, num_gpus_per_node)
            for num_nodes in pipeline_spec_num_nodes
        ]

    def get_num_pipelinespec(self, num_nodes: int) -> List[int]:
        """Return required number of heterogeneous pipelines that
        a linear of combination of the pipelines fills num_nodes.

        Current Alpha-level implementation: always prefer smaller pipelines.
        TODO: analyze the best optimal combination that has the highest throughput.

        Args:
            num_nodes (int): current number of available nodes after failures.

        Returns:
            List[int]: a list of number representing the exact number of
            corresponding pipelines to be deployed.
        """
        if num_nodes > self.max_num_nodes:
            return []

        result = [0] * len(self.pipeline_specs)

        result[0] = math.floor(num_nodes / self.pipeline_specs[0].num_nodes)
        total_assigned_nodes = result[0] * self.pipeline_specs[0].num_nodes
        assert (
            total_assigned_nodes <= num_nodes
        ), f"total assigned nodes {total_assigned_nodes} is not less than total given nodes {num_nodes}"

        smallest_non_zero_pipeline_index = 0
        while total_assigned_nodes < num_nodes:
            while (
                smallest_non_zero_pipeline_index < len(self.pipeline_specs)
                and result[smallest_non_zero_pipeline_index] == 0
            ):
                smallest_non_zero_pipeline_index += 1

            if (
                smallest_non_zero_pipeline_index + 1 < len(self.pipeline_specs)
                and result[smallest_non_zero_pipeline_index] > 0
            ):
                result[smallest_non_zero_pipeline_index] -= 1
                result[smallest_non_zero_pipeline_index + 1] += 1
                total_assigned_nodes += 1

        assert (
            sum(
                result[i] * self.pipeline_specs[i].num_nodes
                for i in range(0, len(self.pipeline_specs))
            )
            == num_nodes
        )

        return result