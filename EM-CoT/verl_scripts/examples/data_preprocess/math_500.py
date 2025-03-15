# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/math500')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'hendrydong/hendrycks_math'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['math500']

    # instruction_following = "Please provide a detailed, step-by-step explanation of your reasoning. Enclose your full thought process within <think> and </think> tags. Finally, present your final answer enclosed in \\boxed{...}."
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            try:
                solution = extract_solution(answer)
            except:
                print(question)
                print(answer)
                raise
            data = {
                "data_source": "DigitalLearningGmbH/MATH-lighteval",
                "prompt": [
                    {"role": "system",
                     "content": system_prompt},
                    {"role": "user",
                     "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn


    def able_to_extract(example):
        
        if example["solution"] is None:
            return False
        if last_boxed_only_string(example["solution"]):
            return True
        return False
    print(len(train_dataset))
    train_dataset = train_dataset.filter(able_to_extract)
    test_dataset = test_dataset.filter(able_to_extract)
    print(len(train_dataset))
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    print((test_dataset))
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    print(test_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
