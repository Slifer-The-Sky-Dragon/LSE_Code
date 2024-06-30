import yaml
from easydict import EasyDict as edict
import os


def load_config(path):
    with open(path, "r", encoding="utf8") as f:
        return edict(yaml.safe_load(f))

def fill_alpha_renyi_parameters(hyper_params):
    if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
        hyper_params["ar_type"] = hyper_params.experiment.regularizers.AlphaRenyi.type
        hyper_params["ar_beta"] = hyper_params.experiment.regularizers.AlphaRenyi.beta
    return hyper_params

def load_hyper_params(config_path, proportion=None, as_reward=False, create_dir=True, train_ratio=None, uniform_noise_alpha=None,
                      gaussian_noise_alpha=None, gamma_noise_beta=None, biased_log_policy=None, disable_weight_decay=None,
                      unbalance=None, gaussian_imbalance=None, data_repeat=None, reward_flip=None, logging_policy_cm=None,
                      propensity_estimation=None, ips_c=None, batch_size=None):
    
    hyper_params = load_config(config_path)
    hyper_params["train_ratio"] = train_ratio
    hyper_params["unbalance"] = unbalance
    hyper_params["gaussian_imbalance"] = gaussian_imbalance
    hyper_params["data_repeat"] = data_repeat
    hyper_params["uniform_noise_alpha"] = uniform_noise_alpha
    hyper_params["gaussian_noise_alpha"] = gaussian_noise_alpha
    hyper_params["biased_log_policy"] = biased_log_policy
    hyper_params["disable_weight_decay"] = disable_weight_decay
    hyper_params["reward_flip"] = reward_flip
    hyper_params["gamma_noise_beta"] = gamma_noise_beta
    hyper_params["logging_policy_cm"] = logging_policy_cm
    hyper_params["propensity_estimation"] = propensity_estimation
    hyper_params["ips_c"] = ips_c
    if batch_size is not None:
        hyper_params["batch_size"] = batch_size


    hyper_params = fill_alpha_renyi_parameters(hyper_params)

    if proportion is not None:
        hyper_params["train_limit"] = int(50_000 * proportion)
        hyper_params["experiment"]["name"] += "_p_" + str(proportion)
    if as_reward:
        hyper_params.experiment.name += "_REWARD"
    common_path = hyper_params["dataset"]
    common_path += f"_{hyper_params.experiment.name}_"
    
    if disable_weight_decay is not None: #We will not use weight decay
        common_path += "_NO&&WEIGHT&&DECAY_"
        hyper_params["weight_decay"] = None
    else:
        common_path += "_wd_" + str(hyper_params["weight_decay"])

        
    common_path += "_lamda_" + str(hyper_params["lamda"])
    
    if (hyper_params.experiment.regularizers is not None) and ("AlphaRenyi" in hyper_params.experiment.regularizers):
        common_path += "_AR_BETA_" + str(hyper_params["ar_beta"])

    if train_ratio is not None:
        common_path += "_TrainRatio_" + str(train_ratio)
    if unbalance is not None:
        common_path += "_Unbalanced_(" + str(unbalance[0]) + "," + str(unbalance[1]) + ")"
    if gaussian_imbalance is not None:
        common_path += f"_ImbalanceGaussian_{gaussian_imbalance}_"
    if data_repeat is not None:
        common_path += "_Rep=" + str(data_repeat) + "_"
    if uniform_noise_alpha is not None:
        common_path += "_UniformNoise_Alpha_" + str(uniform_noise_alpha)
    if gaussian_noise_alpha is not None:
        common_path += "_GaussianNoise_Alpha_" + str(gaussian_noise_alpha)
    if gamma_noise_beta is not None:
        common_path += "_GammaNoise_Beta_" + str(gamma_noise_beta)
    if biased_log_policy is not None:
        common_path += "_BiasedLoggingPolicy"
    if reward_flip is not None:
        common_path += f"_RewardFlip={reward_flip}_"
    if logging_policy_cm is not None:
        splitted_cfm_path = logging_policy_cm.split("/")
        common_path += f"_CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
    if propensity_estimation is not None:
        common_path += "_Propensity_Estimator"
    if batch_size is not None:
        common_path += f"_BSZ={batch_size}"


    if "lse" in hyper_params.experiment.name:
        common_path += "_LSE#Lambda_"
    if "powermean" in hyper_params.experiment.name:
        common_path += "_PowerMean#Lambda_"
    if "exponential_smoothing" in hyper_params.experiment.name:
        common_path += "_ExponentialSmoothing#Alpha_"
    if ips_c is not None:
        common_path += "_IPS#C_" + str(ips_c)

    hyper_params["tensorboard_path"] = "tensorboard_stuff/" + common_path
    hyper_params["log_file"] = "saved_logs/" + common_path
    hyper_params["summary_file"] = "accs/" + common_path
    hyper_params["output_path"] = "models/outputs/" + common_path
    if create_dir: #false in prepare raw dataset
        os.makedirs(os.path.dirname(hyper_params["tensorboard_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["log_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["summary_file"]), exist_ok=True)
        os.makedirs(os.path.dirname(hyper_params["output_path"]), exist_ok=True)
    return hyper_params
