python main_semi_ot.py --config config/fmnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 0.1 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 0.4 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 0.7 --disable_weight_decay
python main_semi_ot.py --config config/fmnist/linear/exponential_smoothing_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --exs_alpha 1 --disable_weight_decay