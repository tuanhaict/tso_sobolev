python main_point.py --parallel --runs_per_gpu 1 --gpus 0 1 2 3
python render.py

# export CUDA_VISIBLE_DEVICES=3

# for l in sw twd; do
#     for id in {0..49}; do
#         python main_tune.py --loss-type $l --source_id $id
#     done
# done