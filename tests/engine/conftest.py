from oobleck_colossalai.pipeline_template import PipelineTemplate

# pylint: disable=unused-argument
from ..conftest import model_name, modules

template_1stage = PipelineTemplate(model_name, [modules])
template_2stages = PipelineTemplate(model_name, [modules[:3], modules[3:]])
template_3stages = PipelineTemplate(
    model_name, [modules[:4], modules[4:7], modules[7:]]
)
