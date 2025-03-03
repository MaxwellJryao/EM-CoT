# CUDA_VISIBLE_DEVICES=0 python stage_1_collect_data.py --local_index=0 --world_size=8 &
# CUDA_VISIBLE_DEVICES=1 python stage_1_collect_data.py --local_index=1 --world_size=8 &
# CUDA_VISIBLE_DEVICES=2 python stage_1_collect_data.py --local_index=2 --world_size=8 &
# CUDA_VISIBLE_DEVICES=3 python stage_1_collect_data.py --local_index=3 --world_size=8 &
# CUDA_VISIBLE_DEVICES=4 python stage_1_collect_data.py --local_index=4 --world_size=8 &
# CUDA_VISIBLE_DEVICES=5 python stage_1_collect_data.py --local_index=5 --world_size=8 &
# CUDA_VISIBLE_DEVICES=6 python stage_1_collect_data.py --local_index=6 --world_size=8 &
# CUDA_VISIBLE_DEVICES=7 python stage_1_collect_data.py --local_index=7 --world_size=8 &

# CUDA_VISIBLE_DEVICES=0 python stage_2_calc_sample_size.py --local_index=0 &
# CUDA_VISIBLE_DEVICES=1 python stage_2_calc_sample_size.py --local_index=1 &
# CUDA_VISIBLE_DEVICES=2 python stage_2_calc_sample_size.py --local_index=2 &
# CUDA_VISIBLE_DEVICES=3 python stage_2_calc_sample_size.py --local_index=3 &
# CUDA_VISIBLE_DEVICES=4 python stage_2_calc_sample_size.py --local_index=4 &
# CUDA_VISIBLE_DEVICES=5 python stage_2_calc_sample_size.py --local_index=5 &
# CUDA_VISIBLE_DEVICES=6 python stage_2_calc_sample_size.py --local_index=6 &
# CUDA_VISIBLE_DEVICES=7 python stage_2_calc_sample_size.py --local_index=7 &

CUDA_VISIBLE_DEVICES=8 python stage_2_sample.py --local_index=0 &
CUDA_VISIBLE_DEVICES=1 python stage_2_sample.py --local_index=1 &
CUDA_VISIBLE_DEVICES=2 python stage_2_sample.py --local_index=2 &
CUDA_VISIBLE_DEVICES=3 python stage_2_sample.py --local_index=3 &
CUDA_VISIBLE_DEVICES=4 python stage_2_sample.py --local_index=4 &
CUDA_VISIBLE_DEVICES=5 python stage_2_sample.py --local_index=5 &
CUDA_VISIBLE_DEVICES=6 python stage_2_sample.py --local_index=6 &
CUDA_VISIBLE_DEVICES=7 python stage_2_sample.py --local_index=7 &