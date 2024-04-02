from dataclasses import dataclass

import simple_parsing
from data_builder import GLUEDataBuilder
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoConfig,
    GPT2ForSequenceClassification,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)

from oobleck import ExecutionEngine, OobleckPlugin


@dataclass
class TrainingArguments:
    model_name_or_path: str = "gpt2"
    global_batch_size: int = 64
    num_epoch: int = 3
    warmup_faction: float = 0.1
    tp_size: int = 1


def main():
    args: TrainingArguments = simple_parsing.parse(TrainingArguments)

    plugin = OobleckPlugin(
        tp_size=args.tp_size,
        global_batch_size=args.global_batch_size,
        microbatch_size=8,
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
        fault_tolerance_threshold=1,
    )

    engine = ExecutionEngine(plugin)

    config: PretrainedConfig = AutoConfig.from_pretrained(args.model_name_or_path)
    config.pad_token_id = config.eos_token_id
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    # Prepare dataloader
    data_builder = GLUEDataBuilder(args.model_name_or_path, plugin, task_name="mrpc")
    dataloader = data_builder.dataloader()

    # optimizer
    optimizer = Adam(model.parameters())

    # lr scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(total_steps * args.warmup_faction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _, _, lr_scheduler = engine.prepare(
        model,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Train model
    model.train()
    optimizer.zero_grad()
    dataloader_iter = iter(dataloader)

    is_pp_last_stage = engine.plugin.stage_manager.is_last_stage()

    for epoch in range(args.num_epoch):
        total_step = len(dataloader)
        dataloader_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not (engine.is_master or is_pp_last_stage),
        ) as pbar:
            for _ in pbar:
                outputs = engine.execute(
                    dataloader_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs.loss,
                    optimizer=optimizer,
                    return_loss=True,
                    return_outputs=True,
                )

                if outputs is None:
                    # Reconfiguration due to failure is done.
                    dataloader_iter = iter(dataloader)
                    continue

                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()


if __name__ == "__main__":
    main()
