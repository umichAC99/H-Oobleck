from cornstarch.pipeline_template import PipelineTemplate

from oobleck.planning.profiler import LayerExecutionResult
from oobleck.planning.ditto import HeteroPipelineTemplate

def create_pipeline_templates(
    model_name: str,
    profile_data: list[LayerExecutionResult],
    num_nodes: list[int],
) -> dict[int, PipelineTemplate]: ...

def create_base_hetero_pipeline_template(
    model_name: str,
    profile_data: list[LayerExecutionResult],
    num_nodes: int,
) -> HeteroPipelineTemplate: ...

def dynamic_programming_recovery(
    node_folding_factors: list[int],
    hetero_cluster: list[int],
    modules_per_stage: list[list[str]],
    layers: list[list[LayerExecutionResult]],
    
) -> HeteroPipelineTemplate: ...