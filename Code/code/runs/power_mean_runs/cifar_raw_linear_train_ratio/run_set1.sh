python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.1 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.3 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.5 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/powermean_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.8 --train_ratio 0.01