from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import utils
import sys
sys.path.append('/scratch/jiarui14/EM-CoT/Online-DPO-R1')
import reward_labeling

# os.environ['CUDA_VISIBLE_DEVICES'] = '8'

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
    start: Optional[int] = field(
        default=0,
        metadata={"help": "Start index"}
    )
    end: Optional[int] = field(
        default=3,
        metadata={"help": "End index"}
    )
    stage_1_samples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples for stage 1 per example"}
    )
    stage_2_samples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples for stage 2 per example"}
    )

# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=42, help='Random seed')
# parser.add_argument('--max_length', type=int, default=2028, help='Max length of newly generated tokens')
# parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Math-7B', help='Model name or path')
# parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
# parser.add_argument('--alpha', type=float, default=0.5, help='Penalty weight alpha')
# parser.add_argument('--beta', type=float, default=2.0, help='Penalty weight beta')
# parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
# script_args = parser.parse_args()

script_args = ScriptArguments()

utils.set_seed(script_args.seed)

ds = load_dataset('HuggingFaceH4/MATH-500')['test']

with open('/scratch/jiarui14/EM-CoT/EM-CoT/data/accept_rates.json', 'r') as f:
    sample_sizes = json.load(f)
with open('/scratch/jiarui14/EM-CoT/EM-CoT/data/accept_rates.json', 'r') as f:
    accept_rates = json.load(f)

script_args.end = min(len(ds), script_args.end)
ds = ds.select(range(script_args.start, script_args.end))
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
llm = LLM(script_args.model_name_or_path, 
        #   gpu_memory_utilization=0.4,
          dtype=torch.bfloat16)

def stage_2_sampling_flat(sample_sizes):
    sampling_params = SamplingParams(
        max_tokens=script_args.max_length,
        temperature=1.0,
        n=1,
    )
    prompts = []
    for i, item in enumerate(ds):
        conv = [{'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for _ in range(sample_sizes[i]):
            prompts.append(conv_chat)
    outputs = llm.generate(prompts, sampling_params)

    idx_sum = 0
    new_outputs = []
    for i in range(len(sample_sizes)):
        new_outputs.append([])
        for idx in range(idx_sum, idx_sum + sample_sizes[i]):
            new_outputs[-1].append(outputs[idx].outputs[0].text)

        idx_sum += sample_sizes[i]

    return new_outputs

def stage_2_sampling(sample_sizes):
    new_outputs = []

    for i in range(len(sample_sizes)):
        if sample_sizes[i] == 0:
            new_outputs.append([])
            continue
        sampling_params = SamplingParams(
            max_tokens=script_args.max_length,
            temperature=1.0,
            n=sample_sizes[i],
            # n=8,
        )
        conv = [{'role': 'user', 'content': ds[i]['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts = [conv_chat]
        outputs = llm.generate(prompts, sampling_params)
        new_outputs.append([output.text for output in outputs[0].outputs])

    return new_outputs

stage_2_outputs = stage_2_sampling(sample_sizes)
stage_2_collected_data = []
corrects_2 = []
total_samples = 0
for i, item in enumerate(ds):
    collected_data = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }
    problem_corrects = []
    for j in range(len(stage_2_outputs[i])):
        # correct = reward_labeling.is_equal(outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        # correct = reward_labeling.is_equal(stage_2_outputs[i][j], item['answer'], dataset_name='math500')
        correct = utils.check_correct(stage_2_outputs[i][j], item['answer'], i)
        if correct:
            problem_corrects.append(j)
            # collected_data['outputs'].append(outputs[i].outputs[j].text)
            collected_data['outputs'].append(stage_2_outputs[i][j])
    corrects_2.append(problem_corrects)
    stage_2_collected_data.append(collected_data)
    total_samples += len(collected_data['outputs'])

print('Total collected samples:', total_samples)
stage_2_collected_data_ds = Dataset.from_list(stage_2_collected_data)
stage_2_collected_data_ds.save_to_disk('data/stage_2_collected_data')

print('done!')