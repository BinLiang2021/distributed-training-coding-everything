""" 
@file_name: train_llama_8b_z2.py
@date: 2024-10-13 
@author: Bin Liang
We use this script to train the Llama-3.1-8B-Instruct model with deepspeed zero stage 2.
What we can set in the deepspeed configuration file are:
- zero_optimization.stage: 2
- dataset 
- loss function 
- model architecture
- optimizer
- Any other training args
"""


import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    
def get_model():
    # 加载预训练模型和分词器
    model_name = "meta-llama/LLaMA-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

model, tokenizer = get_model()

def format_messages(messages):
    formatted_text = ""
    for message in messages:
        if message["role"] == "user":
            formatted_text += f"[HUMAN]: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_text += f"[AI]: {message['content']}\n"
    return formatted_text

def preprocess_function(examples):  
    # 处理每个样本中的消息列表
    contexts = [format_messages(messages) for messages in examples['messages']]
    tokenized_inputs = tokenizer(contexts, padding="max_length", truncation=True, max_length=1024)

    # 将 BatchEncoding 对象转换为字典
    examples.update(tokenized_inputs)

    return examples
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("input_ids").clone()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
def train():

    # 加载训练和评估数据集
    data_files = {
        'train': '/data/home/bin_liang/documents/distributed-training-coding-everything/datasets/ocra/demo_1000.json',
        'eval': '/data/home/bin_liang/documents/distributed-training-coding-everything/datasets/ocra/demo_1000_test.json'
    }

    dataset = load_dataset('json', data_files=data_files)

    # 定义数据集格式
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=500,
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=1e-5,
        num_train_epochs=10,
        deepspeed='ds_zero_3.json',
        fp16=True  # 启用混合精度训练
    )

    # 创建 Trainer 实例
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval']
    )

    # 开始训练
    trainer.train()
    
    
if __name__ == "__main__":

    train()
    