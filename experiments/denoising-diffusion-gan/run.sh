cat << 'EOF' > run.sh
#!/bin/bash

set -u

run_one_experiment() {
    EXP_NAME="$1"
    TRAIN_LOG="$2"
    TEST_LOG="$3"
    GEN_MODE="$4"
    NOISY_MODE="$5"
    LAMBDA="$6"

    echo "========================================"
    echo "=== START TRAIN: $EXP_NAME ==="
    date

    env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=4,5 \
    torchrun --standalone --nproc_per_node=2 train_ddgan.py \
    --dataset cifar10 --exp "$EXP_NAME" \
    --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
    --num_res_blocks 2 --batch_size 256 --num_epoch 1800 \
    --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 \
    --embedding_type positional --use_ema --ema_decay 0.9999 \
    --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 \
    --lazy_reg 15 --loss n_tsw --T 2500 --L 4 \
    --twd_delta 10 --twd_std 0.1 --twd_gen_mode "$GEN_MODE" \
    --ch_mult 1 2 2 2 --noisy_mode "$NOISY_MODE" \
    --lambda_ $LAMBDA --p_agg 1 --save_content \
    --wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" \
    --save_ckpt_every 25 > "$TRAIN_LOG" 2>&1

    TRAIN_EXIT=$?
    echo "=== TRAIN FINISHED: $EXP_NAME | EXIT CODE: $TRAIN_EXIT ==="
    date

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "Train failed for $EXP_NAME, skip test."
        return $TRAIN_EXIT
    fi

    echo "=== START TEST: $EXP_NAME ==="
    date

    env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=4 \
    python test_ddgan.py --dataset cifar10 \
    --exp "$EXP_NAME" \
    --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
    --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 \
    --ch_mult 1 2 2 2 --max_epoch_id 1800 --min_epoch_id 1400 \
    --compute_fid \
    --wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" \
    > "$TEST_LOG" 2>&1

    TEST_EXIT=$?
    echo "=== TEST FINISHED: $EXP_NAME | EXIT CODE: $TEST_EXIT ==="
    date

    return $TEST_EXIT
}

# config 1
run_one_experiment \
    "ddgan_cifar10_n_tsw_interval_raw" \
    "n_tsw.log" \
    "n_tsw_test.log" \
    "gaussian_raw" \
    "interval" \
    0.00001

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Pipeline stopped at config 1."
    exit $EXIT_CODE
fi

# config 2
run_one_experiment \
    "ddgan_cifar10_n_tsw_ball_1e4" \
    "n_tsw.log" \
    "n_tsw_test.log" \
    "gaussian_raw" \
    "ball" \
    0.0001

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Pipeline stopped at config 2."
    exit $EXIT_CODE
fi

# config 3
run_one_experiment \
    "ddgan_cifar10_n_tsw_interval_1e4" \
    "n_tsw.log" \
    "n_tsw_test.log" \
    "gaussian_raw" \
    "interval" \
    0.0001

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Pipeline stopped at config 3."
    exit $EXIT_CODE
fi

echo "=== ALL EXPERIMENTS FINISHED SUCCESSFULLY ==="
date
EOF