import csv
from pathlib import Path

from oobleck_colossalai.shardformer.policies.gpt2 import (
    GPT2Config,
    GPT2ForSequenceClassificationPolicy,
)

from .engine.data_builder import GLUEDataBuilder

config: GPT2Config = GPT2Config.from_pretrained("gpt2")
config.is_decoder = True
config.n_layer = 4
config.num_labels = GLUEDataBuilder.glue_task_num_labels["mrpc"]
config.pad_token_id = config.eos_token_id

modules: list[str] = GPT2ForSequenceClassificationPolicy.get_all_modules(config)
model_name: str = "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification"

tag: str = "test-gpt2"


def init_profile_data(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_index",
                "layer_name",
                "forward",
                "backward",
                "mem_required",
            ],
        )
        writer.writeheader()
        for index, layer_name in enumerate(modules):
            writer.writerow(
                {
                    "layer_index": index,
                    "layer_name": layer_name,
                    "forward": 1.0,
                    "backward": 1.0,
                    "mem_required": 10,
                }
            )
