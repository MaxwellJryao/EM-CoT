from datasets import load_dataset, Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/jiarui/EM-CoT/Online-DPO-R1/iter_dpo_numina_rule_reward/Train1_Qwen_numina_iter1_reward.json')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--select', type=str, default='one')
args = parser.parse_args()

try:
    ds = load_dataset('json', data_files=args.data_path)['train']
except:
    ds = load_dataset(args.data_path)['train']
new_ds = []

for i, item in enumerate(ds):
    rewards = item['rewards']
    max_idx = np.argmax(rewards)
    if rewards[max_idx] <= args.threshold:
        continue
    problem = ds[i]['problem']
    if args.select == 'one':
        # assert problem in item['prompt']
        conv = [
            {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
            {'role': 'user', 'content': problem + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
            {'role': 'assistant', 'content': item['responses'][max_idx]}
        ]
        new_ds.append({
            'conversations': conv
        })
    else:
        for j in range(len(rewards)):
            if rewards[j] > args.threshold:
                # assert problem in item['prompt']
                conv = [
                    {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
                    {'role': 'user', 'content': problem + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
                    {'role': 'assistant', 'content': item['responses'][j]}
                ]
                new_ds.append({
                    'conversations': conv
                })

print('new_ds', len(new_ds))
new_ds = Dataset.from_list(new_ds)
new_ds.save_to_disk(f'data/raft_train_iter{args.iter}_{args.start}_{args.end}')
new_ds.push_to_hub(f'FlippyDora/raft_train_numia_prompt_iter{args.iter}_{args.start}_{args.end}')