from datasets import Dataset
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    train_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of training samples"}
    )
    num_collect_files: Optional[int] = field(
        default=8,
        metadata={"help": "Number of collected files"}
    )
    iter: Optional[int] = field(
        default=1,
        metadata={"help": "the iteration of the experiment"}
    )
    model_prefix: Optional[str] = field(
        default='Qwen7B',
        metadata={"help": "the model prefix"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

all_data = []
for index in range(script_args.num_collect_files):
    with open(f'data/{script_args.model_prefix}/data_{script_args.iter}/stage_2_collected_data_{index}.json', 'r') as f:
        data = json.load(f)
    all_data.extend(data)

print('Total number of problems:', len(all_data))

new_data = []
for i, item in enumerate(all_data):
    for output in item['outputs']:
        new_item = {
            "problem": item['problem'],
            "output": output,
            "answer": item['answer']
        }
        new_data.append(new_item)

print('Total number of samples:', len(new_data))

ds = Dataset.from_list(new_data)

remove_columns = ds.column_names
ds = ds.map(
    lambda x: {
        "conversations": [
            {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
            {'role': 'user', 'content': x['problem'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
            {'role': 'assistant', 'content': x['output']}
        ]
    },
    remove_columns=remove_columns
)

script_args.train_size = min(script_args.train_size, len(ds))
ds = ds.shuffle(seed=script_args.seed).select(range(script_args.train_size))
ds.save_to_disk(f'data/{script_args.model_prefix}/data_{script_args.iter}/train_data')