tau=$1
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL.yaml --tau ${tau} --ul 0 --device cuda:0 --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL.yaml --tau ${tau} --ul 1 --device cuda:0 --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL.yaml --tau ${tau} --ul 4 --device cuda:0 --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL.yaml --tau ${tau} --ul 9 --device cuda:0 --raw_image --linear --feature_size 2048
python main_semi_ot.py --config config/cifar100/linear/ips_bandit_KL.yaml --tau ${tau} --ul 49 --device cuda:0 --raw_image --linear --feature_size 2048