from pathlib import Path

from oobleck_colossalai.pipeline_template import PipelineTemplate

def create_pipeline_templates(
    model_name: str,
    job_profile_dir: Path,
    microbatch_size: int,
    tp_size: int,
    precision: str,
    num_nodes: list[int],
) -> list[PipelineTemplate]: ...
