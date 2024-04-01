import itertools
from pathlib import Path

import pytest
from oobleck_colossalai.pipeline_template import PipelineTemplate

from oobleck.planning import planner

from ..conftest import init_profile_data, model_name, modules, tag

microbatch_size = 1
tp_size = 1
precision = "fp32"


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    profile_dir_path = tmp_path / tag / "profile"
    init_profile_data(
        profile_dir_path,
        tp_size=tp_size,
        microbatch_size=microbatch_size,
        precision=precision,
    )
    return profile_dir_path


def test_error_for_too_large_num_nodes(profile_dir: Path):
    with pytest.raises(RuntimeError):
        planner.create_pipeline_templates(
            model_name=model_name,
            job_profile_dir=profile_dir,
            microbatch_size=microbatch_size,
            tp_size=tp_size,
            precision=precision,
            num_nodes=[len(modules) + 1],
        )


def test_create_pipeline_templates(profile_dir: Path):
    templates: dict[PipelineTemplate] = planner.create_pipeline_templates(
        model_name=model_name,
        job_profile_dir=profile_dir,
        microbatch_size=microbatch_size,
        tp_size=tp_size,
        precision=precision,
        num_nodes=[1, 2, 3, 4],
    )

    assert len(templates) == 4
    assert [template.num_stages for template in templates.values()] == [1, 2, 3, 4]

    for num_stages, template in templates.items():
        assert num_stages == template.num_stages
        assert modules == list(
            itertools.chain.from_iterable(template.modules_per_stage)
        )
