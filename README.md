# **PIE (Perm Is Easy)**

## Installation

In a python environment with pytorch>=2.0 (and cuda, nccl...):
```bash
git clone https://github.com/idriscnrs/bench_pie.git
cd bench_pie
pip install --editable .
```

The code is a finetuning of [the bitext-customer-support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) dataset. So you need to download the dataset with:
```bash
git lfs install
git lfs clone https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
```

The code is compatible with 2 type of fondation models: the [Llama-2 Family](https://huggingface.co/meta-llama) and the [Falcon Family](https://huggingface.co/tiiuae).
Therefore, you need to download the models you want to use. Example with Llama-2 7b:
```bash
git lfs install
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```
It is possible that you need to log in with an HuggingFace Account to download some models.

## Quickstart
You can set the paths of the model weights and the dataset on a config.json like this:
```json
{
	"model_dir": "/work/model_dir/",
	"csv_data_path": "/work/dataset/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
}
```

On a cluster with slurm, you should always use an `ntasks-per-node` equal to the number of gpus in the node. You can do a quick test with the following command:
```bash
srun pie train -c config.json --training_dist 'fsdp' \
--batch_size 1 --seq_length 1024 --debug --model_name "Llama-2-7b-hf"
```

## Benchmarks on Jean-Zay
Libraries version:
 - Torch : 2.1.1
 - Deepspeed : 0.12.4

Hyperparameters :
 - Max Sequence Length = 1024
 - No layer freezed
 - No PEFT
#### Llama 7b :
|n*GPU|Optimized DDP|ZeRO stage|Batch size/gpu|Epoch duration (s)|Training tokens/s|Inference tokens/s|GPU memory allocated(GB)|Avg loss|perplexity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|4|deepspeed|3|2|542.12|9176|12124|56.34|0.53|1.65|
|8|deepspeed|2|3|224.43|21520|53578|64.18|0.57|1.63|
|16|deepspeed|3|4|574.72*|14134*|11029|65.59|0.65|1.65|
|4|fsdp|/|3|394.62*|13648*|23495|52.12|0.63|1.69|
|8|fsdp|/|3|198.73|27371|42343|47.41|0.68|1.72|
|16|fsdp|/|4|450.57|16420|16530|59.10|0.78|1.80|
---
#### Llama 13b :
|n*GPU|Optimized DDP|ZeRO stage|Batch size/gpu|Epoch duration (s)|Training tokens/s|Inference tokens/s|GPU memory allocated(GB)|Avg loss|perplexity|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|8|deepspeed|3|1|621.14|8151|7723|48.51|0.52|1.64|
|16|deepspeed|3|2|1266.39|5065**|3303**|58.93|0.57|1.62|
|4|fsdp|/|2|754.29|6770|10989|62.82|0.58|1.64|
|8|fsdp|/|2|399.42|13531|19310|53.71|0.62|1.66|
|16|fsdp|/|2|1985.68|3707|2378|49.15|0.67|1.70|
---
\* Two configs can have the same training throughput but different epoch duration because the throughput only takes in account, in an iteration, the delta time between the loading of data and the optimizer step whereas the epoch duration which also takes in account an inference part containing the evaluation loop and a test of generating texts.
</br>** The training throughput can be faster than the evaluation throughput because the training only deals with batch size with a length of 1024 whereas the evaluation throughput which deals with, in average, shorter sequence length (e.g. : ~256) so the batch size is less optimized.
