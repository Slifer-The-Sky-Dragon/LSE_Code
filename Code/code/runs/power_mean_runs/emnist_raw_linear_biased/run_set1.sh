python main_semi_ot.py --config config/emnist/linear/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0 --biased_log_policy
python main_semi_ot.py --config config/emnist/linear/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.1 --biased_log_policy
python main_semi_ot.py --config config/emnist/linear/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:1 --raw_image --linear --power_mean_lambda 0.3 --biased_log_policy