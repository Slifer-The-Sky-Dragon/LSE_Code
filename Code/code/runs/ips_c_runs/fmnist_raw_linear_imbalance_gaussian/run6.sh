CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/fmnist/linear/ips_c_bandit_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --ips_c 0.01 --raw_image --linear --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/fmnist/linear/ips_c_bandit_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --ips_c 0.1 --raw_image --linear --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/fmnist/linear/ips_c_bandit_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --ips_c 1 --raw_image --linear --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/fmnist/linear/ips_c_bandit_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --ips_c 10 --raw_image --linear --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/fmnist/linear/ips_c_bandit_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --ips_c 100 --raw_image --linear --disable_weight_decay --gaussian_imbalance 20000