# **PIE (Perm Is Easy)**

## Installation

In working python environment with pytorch>2.0:
```bash
pip install --user --editable .
```

## Quickstart
On a cluster with slurm:
```bash
srun pie train -c config.json --training_dist 'fsdp' \
--batch_size 8 --seq_length 1024 --debug --model_name "meta-llama/Llama-2-7b-hf" --epochs 1
```
