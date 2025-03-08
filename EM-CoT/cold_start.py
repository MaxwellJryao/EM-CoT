from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    data_size: Optional[int] = field(
        default=5000,
        metadata={"help": "Number of training samples"}
    )
    data_path: Optional[str] = field(
        default="nvidia/AceMath-Instruct-Training-Data",
        metadata={"help": "Path to the dataset"}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"}
    )

parser = HfArgumentParser(ScriptArguments)
scripts_args = parser.parse_args_into_dataclasses()[0]

if 'acemath' in scripts_args.data_path.lower():
    ds = load_dataset('parquet', data_files='data/AceMath-Instruct-Training-Data/data/math_sft.parquet')['train']
    ds = ds.shuffle(seed=scripts_args.seed).select(range(scripts_args.data_size))

new_ds = []
for item in ds:
    new_item = {
        'conversations': [
            {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
            {'role': 'user', 'content': item['messages'][0]['content'] + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
            {'role': 'assistant', 'content': item['answer']}
        ]
    }
    new_ds.append(new_item)

new_ds = Dataset.from_list(new_ds)

new_ds.save_to_disk('data/code_start_data')
