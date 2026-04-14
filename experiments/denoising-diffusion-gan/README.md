# Source code for Diffusion Experiments
Based on the code from [RPSW](https://github.com/khainb/RPSW).

## Installation
Python 3.9.12 is used for the experiments. The code is tested on Ubuntu 20.04.1 LTS.

Install the required packages using the following command:
```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install `power_spherical` package:
```bash
cd power_spherical
pip install .
```

No need to setup data for CIFAR-10, as the code will download the dataset automatically.

## CIFAR-10 Training
**Note**: the code only tested on a **single GPU**. Some modifications may be needed for multi-GPU training.

For DDGAN
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 \
    --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
    --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 \
    --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 \
    --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss gan \
    --ch_mult 1 2 2 2 --save_content \
    --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For SW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss sw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```
For EBSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss maxsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For RPSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss rpsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For DSW:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss dsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For EBSW
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ebsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For IWRPSW:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ebrpsw --L 10000 --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For TSW-SL:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 0 --twd_gen_mode gaussian_raw --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For Db-TWD:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 10 --twd_gen_mode gaussian_raw --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For Db-TWD-perp:
```bash
torchrun --standalone --nproc_per_node=4 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss cltwd --T 2500 --L 4 --twd_delta 10 --twd_gen_mode gaussian_orthogonal --ch_mult 1 2 2 2 --save_content --wandb_project_name "twd" --wandb_entity "wandb-userid"
```

For TS-Sobolev:
```bash
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_test --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 32 --num_epoch 3 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ts_sobolev --T 2500 --L 4 --twd_delta 10 --twd_gen_mode gaussian_orthogonal --ts_sobolev_p 2 --ch_mult 1 2 2 2 --save_content --wandb_project_name "ts-gsobolev" --wandb_entity "tuanhaict-" 
```

For GTS-Sobolev
```bash
torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp_squared --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss ts_gsobolev --T 2500 --L 4 --twd_delta 10 --twd_std 0.1 --twd_gen_mode gaussian_raw --ts_sobolev_p 2 --ch_mult 1 2 2 2 --n_function exp_squared --p_agg 2 --save_content --wandb_project_name "ts-gsobolev" --wandb_entity "tuanhaict-" --save_ckpt_every 25
```

```bash
nohup env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=2,4 torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_n_tsw_ball_orthogonal --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss n_tsw --T 2500 --L 4 --twd_delta 10 --twd_std 0.1 --twd_gen_mode gaussian_orthogonal --ch_mult 1 2 2 2 --noisy_mode "ball" --lambda_ 0.00001 --p_agg 1 --save_content --wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" --save_ckpt_every 25 > n_tsw_orthogonal.log 2>&1 &
```
#### CIFAR-10 Testing ####
For testing the trained model, use the name of the experiment in the `--exp` argument. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_exp_squared --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --max_epoch_id 1800 --compute_fid \
--wandb_project_name "ts-gsobolev" --wandb_entity "tuanhaict-"
```

```bash
nohup env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=4 python test_ddgan.py --dataset cifar10 --exp ddgan_cifar10_n_tsw_ball_orthogonal --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --max_epoch_id 1800 --min_epoch_id 1400 --compute_fid \
--wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" > n_tsw_test_orthogonal.log 2>&1 &
```


```bash
nohup env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train_ddgan.py --dataset cifar10 --exp ddgan_cifar10_n_tsw_interval --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 256 --num_epoch 1800 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --loss n_tsw --T 2500 --L 4 --twd_delta 10 --twd_std 0.1 --twd_gen_mode gaussian_raw --ch_mult 1 2 2 2 --noisy_mode "ball" --lambda_ 0.00001 --p_agg 1 --save_content --wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" --save_ckpt_every 25 > n_tsw.log 2>&1 &
```


cat << 'EOF' > run.sh
#!/bin/bash

echo "=== START TRAIN ==="
date

env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc_per_node=2 train_ddgan.py \
--dataset cifar10 --exp ddgan_cifar10_n_tsw_ball_orthogonal \
--num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --batch_size 256 --num_epoch 1800 \
--ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 \
--embedding_type positional --use_ema --ema_decay 0.9999 \
--r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 \
--lazy_reg 15 --loss n_tsw --T 2500 --L 4 \
--twd_delta 10 --twd_std 0.1 --twd_gen_mode gaussian_orthogonal \
--ch_mult 1 2 2 2 --noisy_mode "ball" \
--lambda_ 0.00001 --p_agg 1 --save_content \
--wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" \
--save_ckpt_every 25 > n_tsw_orthogonal.log 2>&1

TRAIN_EXIT=$?
echo "=== TRAIN FINISHED, EXIT CODE: $TRAIN_EXIT ==="
date

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Train failed, do not run test."
    exit $TRAIN_EXIT
fi

echo "=== START TEST ==="
date

env PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=1 \
python test_ddgan.py --dataset cifar10 \
--exp ddgan_cifar10_n_tsw_ball_orthogonal \
--num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
--num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 \
--ch_mult 1 2 2 2 --max_epoch_id 1800 --min_epoch_id 1400 \
--compute_fid \
--wandb_project_name "n-tsw" --wandb_entity "tuanhaict-" \
> n_tsw_test_orthogonal.log 2>&1

TEST_EXIT=$?
echo "=== TEST FINISHED, EXIT CODE: $TEST_EXIT ==="
date

exit $TEST_EXIT
EOF
chmod +x run.sh