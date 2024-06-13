# -*- coding: utf-8 -*-
# Refactored script to divide into functions and a main function to execute everything

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def install_packages():
    """
    Install necessary packages.
    """
    os.system("pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7")

def setup_config():
    """
    Set up configuration parameters.
    """
    config = {
        "dataset_name": "bsalasp/qa-reglamento",
        "model_name": "NousResearch/Llama-2-7b-chat-hf",
        "new_model": "llama-2-7b-reglamento",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "use_nested_quant": False,
        "output_dir": "./results",
        "num_train_epochs": 10,
        "fp16": False,
        "bf16": False,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "max_grad_norm": 0.3,
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "cosine",
        "max_steps": -1,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
    }
    return config

def load_data(dataset_name):
    """
    Load the dataset.
    """
    dataset = load_dataset(dataset_name)
    return dataset

def load_model_and_tokenizer(model_name, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant):
    """
    Load the model and tokenizer with specified configurations.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def train_model(config, model, tokenizer, dataset):
    """
    Train the model with specified configurations and dataset.
    """
    lora_alpha = config["lora_alpha"]
    lora_r = config["lora_r"]
    lora_dropout = config["lora_dropout"]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=tokenizer.model_max_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_train_epochs"],
            fp16=config["fp16"],
            bf16=config["bf16"],
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=config["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            gradient_checkpointing=config["gradient_checkpointing"],
            max_grad_norm=config["max_grad_norm"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            optim=config["optim"],
            lr_scheduler_type=config["lr_scheduler_type"],
            max_steps=config["max_steps"],
            warmup_ratio=config["warmup_ratio"],
            logging_steps=config["logging_steps"],
        )
    )
    
    trainer.train()

def main():
    """
    Main function to execute the script.
    """
    # Install necessary packages
    install_packages()
    
    # Set up configuration
    config = setup_config()
    
    # Load dataset
    dataset = load_data(config["dataset_name"])
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config["model_name"], 
        config["use_4bit"], 
        config["bnb_4bit_compute_dtype"], 
        config["bnb_4bit_quant_type"], 
        config["use_nested_quant"]
    )
    
    # Train the model
    train_model(config, model, tokenizer, dataset)

if __name__ == "__main__":
    main()
