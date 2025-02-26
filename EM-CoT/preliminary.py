from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import torch
import numpy as np
import random
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
import sys
sys.path.append('/scratch/jiarui14/EM-CoT/Online-DPO-R1')
import reward_labeling

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Max length of newly generated tokens"}
    )
    model_name_or_path: Optional[str] = field(
        default='Qwen/Qwen2.5-Math-7B',
        metadata={"help": "Model name or path"}
    )
    epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of epochs"}
    )
    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "Penalty weight alpha"}
    )
    beta: Optional[float] = field(
        default=2.0,
        metadata={"help": "Penalty weight beta"}
    )
    lr: Optional[float] = field(
        default=0.5,
        metadata={"help": "Learning rate"}
    )

parser = HfArgumentParser((ScriptArguments,))
script_args = parser.parse_args_into_dataclasses()[0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(script_args.seed)

# prepare dataset
ds = load_dataset('HuggingFaceH4/MATH-500')['test']
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

# prepare model
llm = LLM(script_args.model_name_or_path)

sampling_params = SamplingParams(
    temperature=1.0,
    n=8,
    max_tokens=script_args.max_length,
    logprobs=1,
)

# generate
conv = [{'role': 'user', 'content': ds[0]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
print(conv_chat)
prompts = [conv_chat]
outputs = llm.generate(prompts, sampling_params)

print(type(outputs))

def get_logprobs_vllm(prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    logprobs = []
    for output in outputs:
        logprobs.append([])
        for item in output.outputs:
            logprobs[-1].append(item.cumulative_logprob)

    return logprobs

def get_uniform_rand(l, r):
    return np.random.uniform(l, r)

print('done!')