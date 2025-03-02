import numpy as np
import random
import torch
import sys
sys.path.append('/scratch/jiarui14/EM-CoT/Online-DPO-R1')
import reward_labeling

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_correct(output, answer, idx, threshold=-1.0, accept_rates=None):
    if reward_labeling.is_equal(output, answer, dataset_name='math500'):
        reward = 1.0
    elif "\\boxed" in output:
        reward = -0.5
    else:
        reward = -1.0

    if reward > threshold:
        if not accept_rates:
            return True
        elif np.random.rand() < accept_rates[idx]:
            return True
        
    return False
