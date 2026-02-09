export CUDA_VISIBLE_DEVICES=1

# Table
python3 main.py -d ssw
python3 main.py -d s3w
python3 main.py -d ri_s3w_1
python3 main.py -d ri_s3w_5
python3 main.py -d ari_s3w
python3 main.py -d stsw --p 1 --delta 50
python3 main.py -d sbsts --p 1.5 --delta 1 --lr 0.05
python3 main.py -d sbsts --p 2 --delta 1 --lr 0.05


# Figure
# python3 plot_loss.py