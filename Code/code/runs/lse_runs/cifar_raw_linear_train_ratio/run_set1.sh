python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -0.1 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda -0.01 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 1 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 10 --train_ratio 0.01
python main_semi_ot.py --config config/cifar/linear_raw/lse_bandit.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 100 --train_ratio 0.01