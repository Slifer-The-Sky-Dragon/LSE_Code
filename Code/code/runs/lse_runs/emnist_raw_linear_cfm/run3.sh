python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 0.1 --disable_weight_decay --logging_policy_cm cfms/emnist/cfm_4.txt 
python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 1 --disable_weight_decay --logging_policy_cm cfms/emnist/cfm_4.txt 
python main_semi_ot.py --config config/emnist/linear/lse_bandit_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --lse_lambda 10 --disable_weight_decay --logging_policy_cm cfms/emnist/cfm_4.txt 