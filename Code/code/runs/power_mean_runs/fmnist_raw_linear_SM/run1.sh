python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_SM_no_wd.yaml --tau 0.5 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_SM_no_wd.yaml --tau 0.5 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.1 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_SM_no_wd.yaml --tau 0.5 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.3 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_SM_no_wd.yaml --tau 0.5 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.5 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/powermean_bandit_SM_no_wd.yaml --tau 0.5 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.8 --disable_weight_decay