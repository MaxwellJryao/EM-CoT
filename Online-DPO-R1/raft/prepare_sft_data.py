from datasets import load_dataset, Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/jiarui/EM-CoT/Online-DPO-R1/iter_dpo_numina_rule_reward/Train1_Qwen_numina_iter1_reward.json')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--iter', type=int, default=1)
args = parser.parse_args()

ds = load_dataset('json', data_files=args.data_path)['train']
new_ds = []

for i, item in enumerate(ds):
    rewards = item['rewards']
    max_idx = np.argmax(rewards)
    problem = ds[i]['problem']
    # assert problem in item['prompt']
    conv = [
        {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{{}}.'},
        {'role': 'user', 'content': problem + f' Let\'s think step by step and output the final answer within \\boxed{{}}'},
        {'role': 'assistant', 'content': item['responses'][max_idx]}
    ]
    new_ds.append({
        'conversations': conv
    })

print('new_ds', len(new_ds))
new_ds = Dataset.from_list(new_ds)
new_ds.save_to_disk(f'data/raft_train_iter{args.iter}_{args.start}_{args.end}')
new_ds.push_to_hub(f'FlippyDora/raft_train_numia_prompt_iter{args.iter}_{args.start}_{args.end}')