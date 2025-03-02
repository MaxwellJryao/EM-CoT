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
    data_path: Optional[str] = field(
        default='FlippyDora/raft1_train_numia_prompt_0-10000',
        metadata={"help": "Path to the dataset"}
    )
    data_split: Optional[str] = field(
        default='train',
        metadata={"help": "Split of the dataset"}
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
        default=8,
        metadata={"help": "Number of samples for stage 2 per example"}
    )
    local_index: Optional[int] = field(
        default=0,
        metadata={"help": "the local index of the agent"},
    )
    world_size: Optional[int] = field(
        default=8,
        metadata={"help": "the total number of the agents"},
    )

# script_args = ScriptArguments()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

utils.set_seed(script_args.seed)

# prepare dataset
ds = load_dataset(script_args.data_path)[script_args.data_split]
script_args.end = min(len(ds), script_args.end)
ds = ds.select(range(script_args.start, script_args.end))
data_size = len(ds)
one_num_share = data_size // script_args.world_size
ds = ds.select(range(one_num_share * script_args.local_index, one_num_share * (script_args.local_index + 1)))

print(f'Local index: {script_args.local_index}, World size: {script_args.world_size}, Data size: {len(ds)}')
print(f'Start: {one_num_share * script_args.local_index}, End: {one_num_share * (script_args.local_index + 1)}')

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

# prepare model for sampling
llm = LLM(script_args.model_name_or_path, 
        #   gpu_memory_utilization=0.4,
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
stage_1_collected_data_all = []
corrects = []
for i, item in enumerate(ds):
    collected_data = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }
    collected_data_all = {
        'problem': item['problem'],
        'answer': item['answer'],
        'outputs': []
    }

    problem_corrects = []
    for j in range(script_args.stage_1_samples):
        # correct = reward_labeling.is_equal(outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        # correct = reward_labeling.is_equal(stage_1_outputs[i].outputs[j].text, item['answer'], dataset_name='math500')
        correct = utils.check_correct(stage_1_outputs[i].outputs[j].text, item['answer'], i)
        if correct:
            problem_corrects.append(j)
            # collected_data['outputs'].append(outputs[i].outputs[j].text)
            collected_data['outputs'].append(stage_1_outputs[i].outputs[j].text)
        collected_data_all['outputs'].append(stage_1_outputs[i].outputs[j].text)
    corrects.append(problem_corrects)
    stage_1_collected_data.append(collected_data)
    stage_1_collected_data_all.append(collected_data_all)

# print(corrects)
# stage_1_collected_data_ds = Dataset.from_list(stage_1_collected_data)
# stage_1_collected_data_ds.save_to_disk(f'data/stage_1_collected_data_{script_args.local_index}')
with open(f'data/stage_1_collected_data_{script_args.local_index}.json', 'w', encoding='utf-8') as f:
    json.dump(stage_1_collected_data, f, indent=4, ensure_ascii=False)

with open(f'data/stage_1_collected_data_all_{script_args.local_index}.json', 'w', encoding='utf-8') as f:
    json.dump(stage_1_collected_data_all, f, indent=4, ensure_ascii=False)