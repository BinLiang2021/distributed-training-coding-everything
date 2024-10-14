""" 
@file_name: hf_2_openai.py
@date: 2024-10-11 
@author: Bin Liang 
我们将使用Hugging Face的transformers库来加载数据集，然后将其转化成 OpenAI 的 messages 格式
"""


import random
import json
import os
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm.auto import tqdm


def set_agrs():
    
    args = ArgumentParser()
    args.add_argument("--dataset", type=str, default="Open-Orca/OpenOrca")
    args.add_argument("--save_folder", type=str, default="datasets")
    args.add_argument("--save_name", type=str, default="ocra")
    args.add_argument("--split", type=str, default="train")
    args = args.parse_args()
    
    return args


def main(args):
    
    os.makedirs(f"{args.save_folder}/{args.save_name}", exist_ok=True)

    
    ds = load_dataset(path=args.dataset, split=args.split)
    random_indices = random.sample(range(len(ds)), 400000)
    
    for i, data in enumerate(tqdm(ds)):
        messages = {
            "messages": [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["response"]}
            ] 
        }
                
        with open(f"{args.save_folder}/{args.save_name}/whole.json", "a") as f:
            f.write(json.dumps(messages))
            f.write("\n")
            
        if i in random_indices:
            with open(f"{args.save_folder}/{args.save_name}/train.json", "a") as f:
                f.write(json.dumps(messages))
                f.write("\n")
        else:
            with open(f"{args.save_folder}/{args.save_name}/test.json", "a") as f:
                f.write(json.dumps(messages))
                f.write("\n")
            
        
    
    
if __name__ == "__main__":
    
    args = set_agrs()
    main(args)
    

