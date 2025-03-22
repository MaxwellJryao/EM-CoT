steps=(10 20 30 40 50 60 70 80 90 100 110 120 130 140)

policy_loss=vanilla
algorithm=raft
base_model=Qwen2.5-Math-1.5B

for step in ${steps[@]}; do
    # em raft
    # actor_dir=checkpoints/em-raft/${base_model}-${algorithm}-${policy_loss}-numina_math_em-sample1n8-sample4-iter1/global_step_$step/actor
    # python scripts/model_merger.py --local_dir=$actor_dir
    # huggingface-cli upload-large-folder ScaleML-RLHF/${base_model}-${algorithm}-${policy_loss}-numina_math_em-sample1n8-sample4-iter1-step_$step --repo-type=model ${actor_dir}/huggingface --num-workers=16

    # raft
    actor_dir=checkpoints/em-raft/${base_model}-${algorithm}-${policy_loss}-numina_math_15_all-n4/global_step_$step/actor
    python scripts/model_merger.py --local_dir=$actor_dir
    huggingface-cli upload-large-folder ScaleML-RLHF/${base_model}-${algorithm}-${policy_loss}-numina_math_15_all-n4-step_$step --repo-type=model ${actor_dir}/huggingface --num-workers=16
done

