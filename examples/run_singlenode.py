import multiprocessing
import os
from dataclasses import asdict, dataclass
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
from typing import Any

import torch
from simple_parsing import ArgumentParser

from oobleck.elastic.agent import Worker
from oobleck.elastic.run import HostInfo

"""
Test Oobleck execution on a single node without using fault tolerance.
"""


@dataclass
class LaunchArguments:
    tag: str
    num_workers: int = 1
    base_dir: Path = Path("/tmp/oobleck")
    script: Path = Path("run_gpt2.py")


@dataclass
class TrainingArguments:
    model_name_or_path: str
    global_batch_size: int = 64
    num_epoch: int = 3
    warmup_faction: float = 0.1
    tp_size: int = 1


def arguments_to_argv(args: dict[str, Any]) -> list[str]:
    argv = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:  # Only add flag if True
                argv.append(f"--{key}")
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


def run():
    parser = ArgumentParser()
    parser.add_arguments(LaunchArguments, dest="launch")
    parser.add_arguments(TrainingArguments, dest="training")

    args = parser.parse_args()
    launch_args: LaunchArguments = args.launch
    training_args: TrainingArguments = args.training

    num_devices = torch.cuda.device_count()
    assert num_devices >= launch_args.num_workers, (
        f"Number of available devices ({num_devices}) is less than "
        f"the number of workers ({launch_args.num_workers})"
    )

    context: SpawnContext = multiprocessing.get_context("spawn")
    processes: list[tuple[SpawnProcess, multiprocessing.Pipe]] = []
    args = arguments_to_argv(asdict(training_args))
    tp_size = training_args.tp_size
    num_agents = launch_args.num_workers // tp_size

    for gpu_index in range(launch_args.num_workers):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        pipe, child_pipe = context.Pipe()
        pipe.send([HostInfo("localhost", tp_size, i) for i in range(num_agents)])

        process = context.Process(
            target=Worker.worker_main,
            args=(
                child_pipe,
                gpu_index // tp_size,
                gpu_index % tp_size,
                launch_args.tag,
                launch_args.base_dir,
                launch_args.script,
                args,
            ),
        )
        process.start()
        processes.append((process, pipe))

    del os.environ["CUDA_VISIBLE_DEVICES"]

    # Rebroadcast pipe information
    port = processes[0][1].recv()
    for _, pipe in processes:
        pipe.send(port)

    for process, _ in processes:
        process.join()


if __name__ == "__main__":
    run()
