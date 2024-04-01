class LayerExecutionResult:
    def __init__(
        self,
        layer_index: int,
        forward: float,
        backward: float,
        allreduce_in_node: dict[int, float],
        allreduce_across_nodes: dict[int, float],
        mem_required: tuple[int, int],
    ): ...
    _index: int
    _forward: float
    _backward: float
    _allreduce_in_node: dict[int, float]
    _allreduce_across_nodes: dict[int, float]
    _mem_required: tuple[int, int]
    
class NodeConfig:
    def __init__(self, node_type: str, num_nodes: int, num_gpus_per_node: int, compute_power: float): ...
    _node_type_idx: int
    _num_nodes: int
    _num_gpus: int
    _compute_power: float
    
class HeteroNodeSpec:
    def get(self) -> list[NodeConfig]: ...
    def at(self, index: int) -> NodeConfig: ...
    def size(self) -> int: ...

class LayerExecutionResults:
    def get(self) -> list[LayerExecutionResult]: ...
    def at(self, index: int) -> LayerExecutionResult: ...
    def size(self) -> int: ...

class StageExecutionResult:
    def __init__(
        self,
        LayerExecutionResults,
        layer_indices: tuple[int, int],
        num_gpus: int,
    ): ...
    _num_gpus: int
    _layer_indices: list[int]
    _size: int
    _mem_required: int

def get_profile_results(
    model_name: str, model_tag: str, microbatch_size: int, node_type = "": str
) -> LayerExecutionResults: ...

def get_hetero_profile_results(
    model_name: str, model_tag: str, microbatch_size: int, node_types:list[str]
) -> list[LayerExecutionResults]: ...

class PipelineTemplate:
    def __init__(
        self,
        stages: list[StageExecutionResult],
        iteration_time: float,
        num_layers: int,
        num_nodes: int,
        num_gpus_per_node: int,
    ): ...
    def get_stages(self) -> list[StageExecutionResult]: ...
    _num_nodes: int
    _num_gpus_per_node: int
    _stages: list[StageExecutionResult]
    _iteration_time: float
    def get_rank_grid(self, ranks: list[int]) -> dict[int, list[int]]: ...
    
class HeteroPipelineTemplate
    def __init__(
        self,
        stages: list[StageExecutionResult],
        num_layers: int,
        node_spec: HeteroNodeSpec,
    ): ...
    def get_stages(self) -> list[StageExecutionResult]: ...
    _stages: list[StageExecutionResult]
    _node_spec: HeteroNodeSpec

class PipelineTemplateGenerator:
    def __init__(self): ...
    def create_pipeline_templates(
        self,
        layer_execution_results: LayerExecutionResults,
        num_nodes: tuple[int, int],
        num_gpus_per_node: int,
    ) -> list[PipelineTemplate]: ...
    def create_hetero_pipeline_template(
        self,
        layer_execution_results: list[LayerExecutionResults],
        node_spec: HeteroNodeSpec,
    ) -> HeteroPipelineTemplate: ...
