import itertools
from pathlib import Path

import pytest
from oobleck_colossalai.pipeline_template import PipelineTemplate

from oobleck.planning import planner

from ..conftest import init_profile_data, model_name, modules, tag

microbatch_size = 1


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    profile_dir_path = tmp_path / tag / "profile"
    init_profile_data(profile_dir_path / f"mb_{microbatch_size}.csv")
    return profile_dir_path


def test_error_for_too_large_num_nodes(profile_dir: Path):
    with pytest.raises(RuntimeError):
        planner.create_pipeline_templates(
            model_name=model_name,
            microbatch_size=microbatch_size,
            num_nodes=[len(modules) + 1],
            job_profile_dir=profile_dir,
        )


def test_create_pipeline_templates(profile_dir: Path):
    templates: list[PipelineTemplate] = planner.create_pipeline_templates(
        model_name=model_name,
        microbatch_size=microbatch_size,
        num_nodes=[1, 2, 3, 4],
        job_profile_dir=profile_dir,
    )

    assert len(templates) == 4
    assert [template.num_stages for template in templates] == [1, 2, 3, 4]

    for template in templates:
        assert modules == list(
            itertools.chain.from_iterable(template.modules_per_stage)
        )
