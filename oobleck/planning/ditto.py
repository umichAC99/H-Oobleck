from cornstarch.pipeline_template import PipelineTemplate

from oobleck.planning.profiler import LayerExecutionResult

class HeteroPipelineTemplate:
    """A temp structure: template for a single pipeline with heterogeneous devices that can be used to instantiate identical pipelines."""
    def __init__(
        self,
        model_name: str,
        modules_per_stage: list[list[str]],
        latency_per_stage: list[float],
        device_type_per_stage: list[int],
        latency: float = 0.0,
        kstar_latency: float = 0.0,
        mem_required: int = 0,
    ):
        self.model_name = model_name
        self.modules_per_stage = modules_per_stage
        self.pseudo_latency = latency  # latency with fake mb (4*num_stages)
        self.kstar_latency = kstar_latency
        self.mem_required = mem_required
        self.device_type_per_stage = device_type_per_stage
        self.latency_per_stage = latency_per_stage

    def __repr__(self) -> str:
        header = f"HeteroPipelineTemplate({self.model_name}, {self.num_stages} stages)"
        body = ""
        for stage, device_type, latency in zip(self.modules_per_stage, self.device_type_per_stage, self.latency_per_stage):
            body += f"\n{device_type}: {stage}({latency})"
        return header + body

    def latency(self, num_microbatches: int) -> float:
        """Return estimated latency with given actual num_microbatches."""
        assert (
            num_microbatches >= self.num_stages
        ), "Numbe rof microbatches must be >= num_stages."
        return self.pseudo_latency + self.kstar_latency * (
            num_microbatches - 4 * self.num_stages
        )
    
    def get_stage_indices(self) -> list[list[int]]:
        """Return indices of the layers in stages in the pipeline."""
        start = 0
        end = 0
        indices = []
        for stage in self.modules_per_stage:
            end += len(stage)
            indices.append(list([start, end-1]))
            start = end
        return indices

    @property
    def num_layers(self) -> int:
        return sum(len(stage) for stage in self.modules_per_stage)

    @property
    def num_stages(self) -> int:
        return len(self.modules_per_stage)

    def __eq__(self, template: PipelineTemplate) -> bool:
        return self.modules_per_stage == template.modules_per_stage

    def __hash__(self) -> int:
        return hash(tuple(tuple(modules) for modules in self.modules_per_stage))

def node_folding(
    profile_data: list[list[LayerExecutionResult]],
    hetero_cluster: list[(str, int)],
):
    """Given profiling results for each device in a hetero cluster, 
    return the total number of normalized nodes based on the ``weakest" device profile
        and node folding factors for each device."""
    # for each device profiling result, get total cost
    profile_data_sum = [
        sum([layer.forward+layer.backward for layer in profile_data[i]])
        for i in range(len(profile_data))
    ]
    # find the device idx with the largest latency
    max_latency_device = 0
    max_latency = 0.0
    for i, device in enumerate(hetero_cluster):
        if profile_data_sum[i] > max_latency:
            max_latency = profile_data_sum[i]
            max_latency_device = i
    
    # find the list of node folding factors
    node_folding_factors = [
        round(profile_data_sum[max_latency_device] / profile_data_sum[i])
        for i in range(len(hetero_cluster))
    ]
    
    total_nodes = sum([
        node_folding_factors[i] * hetero_cluster[i][1]
        for i in range(len(hetero_cluster))
    ])
    
    return total_nodes, node_folding_factors

    

