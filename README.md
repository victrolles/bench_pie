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

On a cluster with slurm you can do a quick test with the following command:
```bash
srun pie train -c config.json --training_dist 'fsdp' \
--batch_size 1 --seq_length 1024 --debug --model_name "Llama-2-7b-hf"
```

## Result on Jean Zay
Llama 7b, deepspeed stage 3, seq_length=1024:

|n*GPU|batch size/gpu|Epoch duration(s)|avg loss|perplexity|GPU memory(GB)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|4|1|841.38|0.531|1.668|41.8|
|8|2|277.12|0.552|1.639|43.79|
|16|4|401.65|0.653|1.645|65.59|
