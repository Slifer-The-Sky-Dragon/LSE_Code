CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01 --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.1 --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 1 --disable_weight_decay --gaussian_imbalance 20000
CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/emnist/linear/lse_bandit_AR_no_wd.yaml --tau 0.1 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 10 --disable_weight_decay --gaussian_imbalance 20000