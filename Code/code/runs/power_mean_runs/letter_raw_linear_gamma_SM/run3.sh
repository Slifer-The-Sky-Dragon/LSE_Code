CUDA_VISIBLE_DEVICES=1 python main_semi_ot.py --config config/letter/linear/powermean_bandit_SM_no_wd.yaml --tau 1.0 --ul 0 --device cuda:0 --raw_image --linear --power_mean_lambda 0.8 --gamma_noise_beta 5.0 --disable_weight_decay