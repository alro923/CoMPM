# python train.py --dataset 'GoEmotions28'
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'MELD'
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'EMORY'
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset 'GoEmotions28'
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'GoEmotions5'

# ln -s '/home/lhj/workspace/data/custom/GoEmotions_5_emotions_txt/original' 'GoEmotions5'
# ln -s '/home/lhj/workspace/data/custom/GoEmotions_28_emotions_txt/original' 'GoEmotions28'

# nvidia-smi
# gpustat -i

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'GoEmotions28' --initial 'scratch' # --lr 1e-6 (default)
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'GoEmotions28' --initial 'scratch' --lr 1e-5
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'GoEmotions28' --initial 'scratch' --lr 1e-3 --epoch 10

# 220817
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'GoEmotions28' --lr 1e-5 --epoch 20


OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'GoEmotions5' --epoch 20

## 0819
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'GoEmotions28_multi' --epoch 10

## todo
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'GoEmotions28_multi' --initial 'scratch' --epoch 20
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'GoEmotions28_multi' --initial 'pretrained' --epoch 20

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'MELD' --epoch 1000 --batch 2


OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train.py & --lr 1e-5 --dataset MELD --epoch 1000
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train.py & --lr 1e-6 --dataset MELD --epoch 1000