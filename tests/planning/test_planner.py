import itertools
from pathlib import Path

import pytest
from cornstarch.pipeline_template import PipelineTemplate

from oobleck.planning import planner
from oobleck.planning.profiler import LayerExecutionResult
from oobleck.planning.ditto import HeteroPipelineTemplate, node_folding

from ..conftest import (
    init_profile_data,
    load_profile_data,
    model_name,
    modules,
    tag,
)

microbatch_size = 1
tp_size = 1
precision = "fp32"


@pytest.fixture
def profile_data(tmp_path: Path) -> list[LayerExecutionResult]:
    profile_dir_path = tmp_path / tag / "profile"
    init_profile_data(
        profile_dir=profile_dir_path,
        tp_size=tp_size,
        microbatch_size=microbatch_size,
        precision=precision,
    )

    return load_profile_data(
        profile_dir=profile_dir_path,
        tp_size=tp_size,
        microbatch_size=microbatch_size,
        precision=precision,
    )
    
@pytest.fixture
def profile_data_hetero_device(tmp_path: Path) -> list[list[LayerExecutionResult]]:
    profile_dir_path = tmp_path / tag / "profile"
    for i in range(3):
        init_profile_data(
            profile_dir=profile_dir_path / str(i),
            tp_size=tp_size,
            microbatch_size=microbatch_size,
            precision=precision,
            default_cost=1.0 + i,
        ) 

    return [load_profile_data(
        profile_dir=profile_dir_path / str(i),
        tp_size=tp_size,
        microbatch_size=microbatch_size,
        precision=precision,
    ) for i in range(3)]


def test_error_for_too_large_num_nodes(profile_data: list[LayerExecutionResult]):
    with pytest.raises(RuntimeError):
        planner.create_pipeline_templates(
            model_name=model_name,
            profile_data=profile_data,
            num_nodes=[len(modules) + 1],
        )


def test_create_pipeline_templates(profile_data: list[LayerExecutionResult]):
    templates: dict[PipelineTemplate] = planner.create_pipeline_templates(
        model_name=model_name,
        profile_data=profile_data,
        num_nodes=[1, 2, 3, 4],
    )

    assert len(templates) == 4
    assert [template.num_stages for template in templates.values()] == [1, 2, 3, 4]

    for num_stages, template in templates.items():
        assert num_stages == template.num_stages
        assert modules == list(
            itertools.chain.from_iterable(template.modules_per_stage)
        )
        print(template)
        print(template.modules_per_stage)
        
def test_create_base_hetero_pipeline_template(profile_data_hetero_device: list[list[LayerExecutionResult]]):
    print("profile_data_hetero_device", profile_data_hetero_device)
    profile_data = profile_data_hetero_device[2]
    hetero_cluster = [("H100", 1), ("A100", 1), ("V100", 2)]
    hetero_cluster_test = [1, 1, 2]
    num_nodes, node_folding_factors = node_folding(profile_data_hetero_device, hetero_cluster)
    template = planner.create_base_hetero_pipeline_template(
        model_name=model_name,
        profile_data=profile_data,
        num_nodes= num_nodes,
    )

    assert template.num_stages == num_nodes

    print(template)
    print(template.modules_per_stage)
    print(node_folding_factors)
    
    # Ensure that 'template' is an instance of 'HeteroPipelineTemplate'
    assert isinstance(template, HeteroPipelineTemplate)
    
    template = planner.dynamic_programming_recovery(node_folding_factors, hetero_cluster_test, template.get_stage_indices(), profile_data_hetero_device)
    print("After recovery:")
    print(template)
