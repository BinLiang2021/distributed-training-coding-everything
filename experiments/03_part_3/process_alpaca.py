import datasets

def alpaca_map_fn(example):
    if example.get('output') == '<nooutput>':
        return None
    elif len(example['input']) > 0:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"{example['instruction']}\n{example['input']}",
                },
                {
                    "role": "assistant",
                    "content": example['output'],
                }
            ]
        }
    else:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": example['instruction'],
                },
                {
                    "role": "assistant",
                    "content": example['output'],
                }
            ]
        }

if __name__ == "__main__":
    alpaca_clean = datasets.load_dataset("yahma/alpaca-cleaned")
    alpaca_clean = alpaca_clean.map(alpaca_map_fn)
    alpaca_clean = alpaca_clean.remove_columns(["input", "instruction", "output"])
    alpaca_clean.save_to_disk("./alpaca-cleaned-processed")
    print( alpaca_clean['train'][0] )