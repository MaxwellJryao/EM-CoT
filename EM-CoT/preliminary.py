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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(script_args.seed)

# prepare dataset
ds = load_dataset('HuggingFaceH4/MATH-500')['test']
script_args.end = min(len(ds), script_args.end)
ds = ds.select(range(script_args.start, script_args.end))
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

# prepare model for sampling
llm = LLM(script_args.model_name_or_path, 
          gpu_memory_utilization=0.4,
          dtype=torch.bfloat16)

def stage_1_sampling():
    sampling_params = SamplingParams(
        max_tokens=script_args.max_length,
        temperature=1.0,
        n=script_args.stage_1_samples,
    )
    prompts = []
    for i, item in enumerate(ds):
        conv = [{'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompts.append(conv_chat)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

stage_1_outputs = stage_1_sampling()

#TODO: currently, stage 1 selects all outputs with correct answers
stage_1_collected_data = []
corrects = []
for i, item in enumerate(ds):
    collected_data = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }
    problem_corrects = []
    for j in range(script_args.stage_1_samples):
        # correct = reward_labeling.is_equal(outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        correct = reward_labeling.is_equal(stage_1_outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        if correct:
            problem_corrects.append(j)
            # collected_data['outputs'].append(outputs[i].outputs[j].text)
            collected_data['outputs'].append(stage_1_outputs[i].outputs[j].text)
    corrects.append(problem_corrects)
    stage_1_collected_data.append(collected_data)

print(corrects)
stage_1_collected_data_ds = Dataset.from_list(stage_1_collected_data)
stage_1_collected_data_ds.save_to_disk('data/stage_1_collected_data')

# calculate the accept rate from stage 1

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
# model.to(torch.device('cuda:8'))
model.cuda()

def calc_grad():
    all_grads = []
    for i, item in enumerate(tqdm(stage_1_collected_data, desc='Calculating gradients')):
        if len(item['outputs']) == 0:
            mean_grad = 0
        else:
            grads = []
            for output in item['outputs']:
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
                loss.backward()
                grad_norm = torch.norm(model.lm_head.weight.grad, p=2).item()
                grads.append(grad_norm)
                model.zero_grad()

            mean_grad = np.mean(grads)
        all_grads.append(mean_grad)

    return all_grads

# calculate accept rates and sample sizes
all_grads = calc_grad()
accept_rates = calc_accept_rate()
sample_sizes = calc_sample_ratio(all_grads, accept_rates)

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

print('Sample sizes:', sample_sizes)

def stage_2_sampling(sample_sizes):
    sampling_params = SamplingParams(
        max_tokens=script_args.max_length,
        temperature=1.0,
        n=1,
    )
    prompts = []
    for i, item in enumerate(ds):
        conv = [{'role': 'user', 'content': item['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}]
        conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for _ in sample_sizes[i]:
            prompts.append(conv_chat)
    outputs = llm.generate(prompts, sampling_params)

    idx_sum = 0
    new_outputs = []
    for i in range(len(sample_sizes)):
        new_outputs.append([])
        for idx in range(idx_sum, idx_sum + sample_sizes[i]):
            new_outputs[-1].append(outputs[idx].outputs[0].text)

        idx_sum += sample_sizes[i]

    return outputs

stage_2_outputs = stage_2_sampling(sample_sizes)
stage_2_collected_data = []
corrects_2 = []
for i, item in enumerate(ds):
    collected_data = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }
    problem_corrects = []
    for j in range(len(stage_2_outputs[i])):
        # correct = reward_labeling.is_equal(outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        correct = reward_labeling.is_equal(stage_2_outputs[i][j], item['answer'], dataset_name='math500')
        if correct:
            problem_corrects.append(j)
            # collected_data['outputs'].append(outputs[i].outputs[j].text)
            collected_data['outputs'].append(stage_2_outputs[i][j])
    corrects_2.append(problem_corrects)
    stage_2_collected_data.append(collected_data)

print('done!')