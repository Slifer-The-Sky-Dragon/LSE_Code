python main_semi_rec2.py --config config/opd/all/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.01  --disable_weight_decay
python main_semi_rec2.py --config config/opd/all/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.1  --disable_weight_decay
python main_semi_rec2.py --config config/opd/all/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 1.0  --disable_weight_decay
python main_semi_rec2.py --config config/opd/all/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 10.0  --disable_weight_decay
