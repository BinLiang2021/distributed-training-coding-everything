import sys
sys.path.append("../")
from exp01.utils import create_datasets
import datasets
from transformers import AutoTokenizer

class DataArgs:
    splits = "train"
    dataset_name = "/home/a100user/llm_train_eval/experiments/exp05_alpaca/alpaca-cleaned-processed"

data_args = DataArgs()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
train_dataset, eval_dataset = create_datasets(
    tokenizer,
    data_args,
    None,
    apply_chat_template=True,
)

print (train_dataset[0])

def tokenize_map_fn(example):
    return {
        "length": len(tokenizer(example["content"])["input_ids"]),
    }

train_dataset = train_dataset.map(tokenize_map_fn, batched=False, num_proc=10)
print (train_dataset[0])
print ("max input size:", max(train_dataset["length"]))

# filter out examples that are too long
train_dataset = train_dataset.filter(lambda x: x["length"] <= 512)
print (train_dataset)

train_dataset = train_dataset.remove_columns("length")

dataset = datasets.DatasetDict({
    "train": train_dataset,
})
print (dataset)
dataset.save_to_disk("./alpaca-cleaned-template-512")
