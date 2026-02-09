export CUDA_VISIBLE_DEVICES=3

python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.005 --delta 1 --p_tsw 1 --p_sobolev 1.2 \
    --dataset_name "8gaussians" --std 0.001 --num_seeds 5

python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.005 --delta 1 --p_tsw 1 --p_sobolev 1.5 \
    --dataset_name "8gaussians" --std 0.001 --num_seeds 5 --eval_sb

python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.005 --delta 1 --p_tsw 1 --p_sobolev 2 \
    --dataset_name "8gaussians" --std 0.001 --num_seeds 5 --eval_sb


python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.05 --delta 1 --p_tsw 1 --p_sobolev 1.2 \
    --dataset_name "gaussian_30d_small_v" --std 0.001 --num_seeds 5

python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.05 --delta 10 --p_tsw 1 --p_sobolev 1.5 \
    --dataset_name "gaussian_30d_small_v" --std 0.001 --num_seeds 5 --eval_sb

python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_sw 0.005 \
    --lr_tsw_sl 0.05 --delta 1 --p_tsw 1 --p_sobolev 2 \
    --dataset_name "gaussian_30d_small_v" --std 0.001 --num_seeds 5 --eval_sb
