tau=$1
python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:1 --raw_image --linear --exs_alpha 0.1 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:1 --raw_image --linear --exs_alpha 0.4 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:1 --raw_image --linear --exs_alpha 0.7 --disable_weight_decay
python main_semi_ot.py --config config/emnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau ${tau} --ul 0 --device cuda:1 --raw_image --linear --exs_alpha 1 --disable_weight_decay