# PPO Training

We implement the PPO training by veRL package and we provide the script heew. Please follow [veRL's document](https://verl.readthedocs.io/en/latest/start/install.html) to set up the training environment. 

## Data prepration of the numia prompt set

First, we need to move the numia_process.py to the verl/examples/data_preprocess/ folder. Then, run the data prepration script by:
```sh 
python verl/examples/data_preprocess/numia_process.py
```
This will prepare the training set and validation set to parquet format.

## Running PPO training with numina prompt set
Similarly, now we move the verl_example.sh to the examples/ppo_trainer/ folder. Then, we need to first set up the environment by 

```sh
export VLLM_ATTENTION_BACKEND=XFORMERS
```
Otherwise, we may encounter the illegal memory error. Then, we can start the PPO training by

```
bash examples/ppo_trainer/verl.sh
```

