python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 0.2 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.1 --unbalance 5 0.6 --data_repeat 2 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 0.2 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.3 --unbalance 5 0.6 --data_repeat 2 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 0.2 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.5 --unbalance 5 0.6 --data_repeat 2 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/powermean_bandit_no_wd.yaml --tau 0.2 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.8 --unbalance 5 0.6 --data_repeat 2 --disable_weight_decay