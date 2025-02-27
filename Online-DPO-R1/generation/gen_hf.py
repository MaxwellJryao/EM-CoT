#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="hendrycks/competition_math",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="problem",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_end: Optional[int] = field(
        default=100,
        metadata={"help": "the size of the dataset"},
    )
    dataset_start: Optional[int] = field(
        default=0,
        metadata={"help": "the start index of the dataset"},
    )
    data_shuffle_seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed for shuffling the dataset"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    #max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
    # gpu_memory_utilization=0.35
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    #stop=["<|user|>"],
)


ds = load_dataset(script_args.dataset_name_or_path, split="train")
if script_args:
    if script_args.dataset_end > 0:
        script_args.dataset_end = min(script_args.dataset_end, len(ds))
    else:
        script_args.dataset_end = len(ds)
    ds = ds.shuffle(seed=script_args.data_shuffle_seed).select(range(script_args.dataset_start, script_args.dataset_end))

# ## loading for MATH training set
# configs = get_dataset_config_names(script_args.dataset_name_or_path)
# datasets = [load_dataset(script_args.dataset_name_or_path, config, split='train') for config in configs]
# ds = concatenate_datasets(datasets)

# ds = ds.remove_columns(["prompt"])

ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(
            [{"role":"user","content":x[script_args.dataset_key] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'}], 
            tokenize=False, add_generation_prompt=True),
        "problem": x[script_args.dataset_key],
    }
)

data_size = len(ds["prompt"])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
print(ds, script_args.dataset_name_or_path)
print(ds[0])


prompts = ds["prompt"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    #tmp_data = {"prompt": ds[i]['prompt'], "responses": [out.text for out in output.outputs], "gt":remove_boxed(last_boxed_only_string(ds[i]['solution']))}
    if 'numia_prompt' in script_args.dataset_name_or_path:
        tmp_data = {"prompt": ds[i]['prompt'], "responses": [out.text for out in output.outputs], "gt":ds[i]['reward_model']['ground_truth'], "problem": ds[i]['problem']}
    else:
        tmp_data = {"prompt": ds[i]['prompt'], "responses": [out.text for out in output.outputs], "gt":ds[i]['gt'], "problem": ds[i]['problem']}
    gathered_data.append(tmp_data)


print("I collect ", len(gathered_data), "samples")


with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')
