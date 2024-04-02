import multiprocessing
import os
from argparse import REMAINDER
from dataclasses import dataclass
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path
from typing import Any

import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers import field

from oobleck.elastic.agent import Worker
from oobleck.elastic.run import HostInfo

"""
Test Oobleck execution on a single node without using fault tolerance.
"""


@dataclass
class LaunchArguments:
    # A tag to identify this run
    tag: str
    # Number of local agents to launch
    num_agents: int = 1
    # Number of GPUs per agent. It should be equal to tp_size in the example script.
    num_gpus_per_agent: int = 1
    # Oobleck root directory to store profiles.
    base_dir: Path = Path("/tmp/oobleck")


@dataclass
class ScriptArguments:
    training_script: Path = field(positional=True)
    training_script_args: list[str] = field(positional=True, nargs=REMAINDER)


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

    parser.add_argument(
        "training_script",
        type=Path,
        help="Full path to the training script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )
    parser.add_argument("training_script_args", nargs=REMAINDER)

    args = parser.parse_args()
    launch_args: LaunchArguments = args.launch
    script_args = ScriptArguments(args.training_script, args.training_script_args)

    num_devices = torch.cuda.device_count()
    num_gpus = launch_args.num_agents * launch_args.num_gpus_per_agent
    assert num_devices >= num_gpus, (
        f"Number of available devices ({num_devices}) is less than "
        f"the number of workers ({num_gpus})"
    )

    context: SpawnContext = multiprocessing.get_context("spawn")
    processes: list[tuple[SpawnProcess, multiprocessing.Pipe]] = []

    dist_info: list[HostInfo] = [
        HostInfo("localhost", launch_args.num_gpus_per_agent, i)
        for i in range(launch_args.num_agents)
    ]

    gpu_index = 0
    for agent_index in range(launch_args.num_agents):
        for worker_index in range(launch_args.num_gpus_per_agent):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            pipe, child_pipe = context.Pipe()
            pipe.send(dist_info)

            process = context.Process(
                target=Worker.worker_main,
                args=(
                    child_pipe,
                    agent_index,
                    worker_index,
                    launch_args.tag,
                    launch_args.base_dir,
                    script_args.training_script,
                    script_args.training_script_args,
                ),
            )
            process.start()
            processes.append((process, pipe))

            gpu_index += 1

    del os.environ["CUDA_VISIBLE_DEVICES"]

    # Rebroadcast pipe information
    port = processes[0][1].recv()
    for _, pipe in processes:
        pipe.send(port)

    for process, _ in processes:
        process.join()


if __name__ == "__main__":
    run()
