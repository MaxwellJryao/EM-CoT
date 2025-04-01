import json
import numpy as np
import pandas as pd
from tqdm import tqdm

ds_names = ['math500', 'minerva_math', 'olympiad_bench', 'aime24', 'amc23']
df = pd.DataFrame(columns=ds_names)

for step in tqdm(range(1, 2, 1)):
    avg_acc = 0
    # for ds in ['math500', 'minerva_math', 'olympiad_bench']:
    step_accs = []
    for ds in ds_names:
        try:
            with open(f'/shared/storage-01/jiarui14/EM-CoT/verl/eval/result/Qwen2.5-Math-1.5B-raft-plusplus-numina_math_em-sample1n32-sample32-iter3-n8_t1.0/{ds}_outputs.json') as f:
                res = json.load(f)
        except:
            # with open(f'/shared/storage-01/jiarui14/EM-CoT/verl/eval/result/Qwen2.5-Math-1.5B-raft-plusplus-numina_math_15_all-n4-step{step}-n8_t1.0/{ds}_outputs.json') as f:
            #     res = json.load(f)
            continue

        acc = 0
        for item in res:
            acc += np.mean(item['scores'])

        step_accs.append(acc / len(res))

    try:
        df.loc[step] = step_accs
    except:
        continue

df['3 average'] = df.iloc[:, :3].mean(axis=1)
df['5 average'] = df.iloc[:, :5].mean(axis=1)
df = df[['math500', 'minerva_math', 'olympiad_bench', '3 average', 'aime24', 'amc23', '5 average']]
print(df)
df.to_csv('res.csv', index=False, float_format='%.4f', sep='\t')
