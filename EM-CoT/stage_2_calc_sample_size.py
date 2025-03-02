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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_length: Optional[int] = field(
        default=4096,
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
        default=100000,
        metadata={"help": "End index"}
    )
    stage_1_samples: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples for stage 1 per example"}
    )
    stage_2_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of samples for stage 2 per example"}
    )
    local_index: Optional[int] = field(
        default=2,
        metadata={"help": "the local index of the agent"}
    )

# script_args = ScriptArguments()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

utils.set_seed(script_args.seed)

# stage_1_collected_data = load_from_disk('data/stage_1_collected_data')
with open(f'/scratch/jiarui14/EM-CoT/EM-CoT/data/stage_1_collected_data_{script_args.local_index}.json', 'r') as f:
    stage_1_collected_data = json.load(f)

script_args.end = min(script_args.end, len(stage_1_collected_data))
stage_1_collected_data = stage_1_collected_data[script_args.start:script_args.end]
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

def calc_accept_rate():
    accept_rates = []
    for item in stage_1_collected_data:
        accept_rate = len(item['outputs']) / script_args.stage_1_samples
        accept_rates.append(accept_rate)
    return accept_rates

def calc_sample_ratio(Gs, ps):
    # Gs: list of gradients
    # ps: list of accept rates
    sample_sizes = []
    for G, p in zip(Gs, ps):
        if G == 0 or p == 0:
            sample_size = 0
        else:
            sample_size = G / (np.sqrt(p + script_args.alpha / np.power(p, script_args.beta - 1.0)))    
        sample_sizes.append(sample_size)
    total = sum(sample_sizes)
    sample_sizes = [sample_size / total for sample_size in sample_sizes]
    return sample_sizes

def find_prompt_end(input_ids):
    end = tokenizer('<|im_start|>assistant\n')['input_ids']
    end_len = len(end)
    input_len = len(input_ids)
    for i in range(input_len - end_len):
        found = True
        for j in range(end_len):
            if input_ids[i + j] != end[j]:
                found = False
                break
        if found:
            return i + end_len
    
    raise ValueError('End not found')
    
# load model for gradient calculation
model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16)
#TODO: currently only use the gradients of lm_head for gradient calculation
for n, p in model.named_parameters():
    if 'lm_head' not in n:
        p.requires_grad = False
params = [p for p in model.parameters() if p.requires_grad]
# model.to(torch.device('cuda:8'))
model.cuda()

def calc_grad():
    all_grads = []
    for i, item in enumerate(tqdm(stage_1_collected_data, desc='Calculating gradients')):
        if i != 1030:
            continue
        if len(item['outputs']) == 0:
            mean_grad = 0
        else:
            grads = []
            for j, output in enumerate(item['outputs']):
                if j != 4:
                    continue
                conv = [
                    {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
                    {'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}
                ]
                conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                conv_chat += output
                input_ids = tokenizer(conv_chat, return_tensors='pt').input_ids.to(model.device)
                o = model(input_ids, output_hidden_states=True)
                logits = o.logits
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                resp_start = find_prompt_end(input_ids[0].tolist())
                output_log_probs = log_probs[0, resp_start:]
                output_log_probs_sen = output_log_probs.sum(dim=0)
                
                # get the gradient by loss backpropagation
                loss = -output_log_probs_sen.mean() / (len(input_ids[0]) - resp_start)
                # loss.backward()
                # grad_norm = torch.norm(model.lm_head.weight.grad, p=2).item()
                gradients = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)[0]
                grad_norm = torch.norm(gradients, p=2).item()
                grads.append(grad_norm)
                model.zero_grad()
                torch.cuda.empty_cache()

            mean_grad = np.mean(grads)
        all_grads.append(mean_grad)

    return all_grads

# calculate accept rates and sample sizes
all_grads = calc_grad()
accept_rates = calc_accept_rate()

with open(f'data/accept_rates_{script_args.local_index}.json', 'w') as f:
    json.dump(accept_rates, f, indent=4)

sample_sizes = calc_sample_ratio(all_grads, accept_rates)

with open(f'data/sample_sizes_ratio_{script_args.local_index}.json', 'w') as f:
    json.dump(sample_sizes, f, indent=4)

def float_to_int_preserve_sum(arr, N):
    # 1. 初步缩放并四舍五入
    scaled_arr = np.array(arr) * N
    int_arr = np.round(scaled_arr).astype(int)
    print(int_arr)

    # 2. 计算误差
    error = N - np.sum(int_arr)

    # 3. 误差修正：根据四舍五入前的误差最小调整
    if error != 0:
        # 计算原始浮点数和转换后整数的误差
        residuals = scaled_arr - int_arr
        # 按误差绝对值最大调整
        indices = np.argsort(-residuals if error > 0 else residuals)[:abs(error)]
        int_arr[indices] += np.sign(error)  # 调整以匹配总和

    return int_arr.tolist()

sample_sizes = float_to_int_preserve_sum(sample_sizes, script_args.stage_2_samples)

with open(f'data/sample_sizes_{script_args.local_index}.json', 'w') as f:
    json.dump(sample_sizes, f, indent=4)

# print('Sample sizes:', sample_sizes)
print('done!')