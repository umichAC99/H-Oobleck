# How to run

You can run Oobleck examples in two ways: run in standalone (using one node without fault tolerance),
or across multiple nodes with fault tolerance capability.

## Run in standalone

> For more configurable arguments, use `python run_singlenode.py --help`. Following example includes minimial necessary information.

Use `run_singlenode.py` script. If you want to run `run_gpt2.py`:
```bash
python run_singlenode.py --tag <tag> --script run_gpt2.py --num_workers 4 --model_name_or_path gpt2
```
where `--model_name_or_path` is the model name in HuggingFace (including company name) to download the model from HuggingFace model.

Oobleck profiles the given model and uses the profile result to generate pipeline templates. The profile result is cached in `<base_dir>/profile/<tag>`.