import torch
import numpy as np
from tqdm import tqdm

from utils import *
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from utils import create_tensors


class DataLoader:
    def __init__(self, hyper_params, x, delta, prop=None, action=None, labeled=None):
        self.x = x
        self.delta = delta
        self.prop = prop
        self.action = action
        self.labeled = labeled
        self.bsz = hyper_params["batch_size"]
        self.hyper_params = hyper_params
        self.dataset = hyper_params["dataset"]

    def __len__(self):
        return len(self.x)

    def __iter__(self, eval=False):
        x_batch, y_batch, action, delta, all_delta, prop, all_prop, labeled = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        data_done = 0
        for ind in tqdm(range(len(self.x))):
            # if self.prop[ind][self.action[ind]] < 0.001:
            #     continue  # Overflow issues, Sanity check
            c, h, w = self.dataset["data_shape"]
            if (
                "raw_image" not in self.hyper_params
                or not self.hyper_params["raw_image"]
            ):
                new_x = self.x[ind].reshape(c, h, w)
                if c == 1:
                    new_x = np.repeat(new_x, repeats=3, axis=0)
            else:
                new_x = self.x[ind]
            # new_x = (new_x - np.array([[[0.485]], [[0.456]], [[0.406]]])) / np.array(
            #     [[[0.229]], [[0.224]], [[0.225]]]
            # )
            x_batch.append(new_x)
            y_batch.append(np.argmax(self.delta[ind]))

            # Pick already chosen action
            choice = self.action[ind]
            action.append(choice)

            delta.append(self.delta[ind][choice])

            if self.prop[ind][choice] < 0.001:
                prop.append(0.001)
            else:
                prop.append(self.prop[ind][choice])

            all_delta.append(self.delta[ind])
            all_prop.append(self.prop[ind])
            if self.labeled is not None:
                labeled.append(self.labeled[ind])
            data_done += 1

            if len(x_batch) == self.bsz:
                if eval == False:
                    if self.labeled is None:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        )
                    else:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), torch.tensor(
                            labeled
                        ).float()
                else:
                    if self.labeled in None:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), all_prop, all_delta
                    else:
                        yield torch.tensor(np.stack(x_batch)).float(), torch.tensor(
                            y_batch, dtype=torch.int64
                        ), torch.tensor(action, dtype=torch.int64), torch.tensor(
                            delta, dtype=torch.float32
                        ), torch.tensor(
                            prop, dtype=torch.float32
                        ), all_prop, all_delta, torch.tensor(
                            labeled
                        ).float()

                x_batch, y_batch, action, delta, all_delta, prop, all_prop, labeled = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )


def readfile(path, hyper_params):
    if "feature_size" in hyper_params:
        feature_size = hyper_params["feature_size"]
    else:
        feature_size = np.prod(hyper_params["dataset"]["data_shape"])
    x, delta, prop, action = [], [], [], []

    data = load_obj(path)
    C = hyper_params["dataset"]["num_classes"]
    print("Loaded data address and shape: ", path, data.shape, C)
    assert (
        data.shape[1] == feature_size + 2 * C + 1
    ), "Data is unlabeled, but fully labeled loader is used."

    for line in data:
        x.append(line[:feature_size])
        delta.append(line[feature_size : (feature_size + C)])
        prop.append(line[(feature_size + C) : (feature_size + 2 * C)])
        action.append(int(line[-1]))

    return np.array(x), np.array(delta), np.array(prop), np.array(action)


def readfile_unlabeled(path, hyper_params):
    if "feature_size" in hyper_params:
        feature_size = hyper_params["feature_size"]
    else:
        feature_size = np.prod(hyper_params["dataset"]["data_shape"])
    x, delta, prop, action, labeled, flip = [], [], [], [], [], []

    data = load_obj(path)
    C = hyper_params["dataset"]["num_classes"]
    # print("Loaded data address and shape: ", path, data.shape, C)
    print(data.shape, feature_size)
    if hyper_params["reward_flip"] is not None:
        assert (
            data.shape[1] == feature_size + 2 * C + 2 + 1
        ), "Data is fully labeled, but unlabeled loader is used. (Warning: Reward flipping has been used)"
    else:
        assert (
            data.shape[1] == feature_size + 2 * C + 2
        ), "Data is fully labeled, but unlabeled loader is used."

    if hyper_params["reward_flip"] is not None:
       for line in data:
            x.append(line[:feature_size])
            delta.append(line[feature_size : (feature_size + C)])
            prop.append(line[(feature_size + C) : (feature_size + 2 * C)])
            action.append(int(line[-3]))
            labeled.append(int(line[-2]))
            flip.append(int(line[-1]))
    else:
        for line in data:
            x.append(line[:feature_size])
            delta.append(line[feature_size : (feature_size + C)])
            prop.append(line[(feature_size + C) : (feature_size + 2 * C)])
            action.append(int(line[-2]))
            labeled.append(int(line[-1]))
            flip.append(0)

    return (
        np.array(x),
        np.array(delta),
        np.array(prop),
        np.array(action),
        np.array(labeled),
        np.array(flip),
    )


def load_data(hyper_params, labeled=True):
    store_folder = hyper_params.dataset["name"]
    path = f"../data/{store_folder}/bandit_data"
    path += "_sampled_" + str(hyper_params["num_sample"])
    print(store_folder)
    print(labeled)
    labeled_train = None
    labeled_val = None
    if labeled:
        x_train, delta_train, prop_train, action_train = readfile(
            path + "_train", hyper_params
        )
        x_val, delta_val, prop_val, action_val = readfile(path + "_val", hyper_params)
    else:
        (
            x_train,
            delta_train,
            prop_train,
            action_train,
            labeled_train,
        ) = readfile_unlabeled(path + "_train", hyper_params)
        x_val, delta_val, prop_val, action_val, labeled_val = readfile_unlabeled(
            path + "_val", hyper_params
        )
    x_test, delta_test, prop_test, action_test = readfile(path + "_test", hyper_params)
    # Shuffle train set
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    delta_train = delta_train[indices]
    prop_train = prop_train[indices]
    action_train = action_train[indices]
    if not labeled:
        labeled_train = labeled_train[indices]

    trainloader = DataLoader(
        hyper_params,
        x_train,
        delta_train,
        prop_train,
        action_train,
        labeled_train if not labeled else None,
    )
    testloader = DataLoader(hyper_params, x_test, delta_test, prop_test, action_test)
    valloader = DataLoader(
        hyper_params, x_val, delta_val, prop_val, action_val, labeled_val
    )

    return trainloader, testloader, valloader


def load_data_fast(hyper_params, device, labeled=True, create_dataset=False):
    if (hyper_params["propensity_estimation"] is not None and create_dataset==True) \
            or (hyper_params["gaussian_imbalance"] is not None and create_dataset==True):
        dataset = hyper_params["dataset_name_string"]
        tau_value = hyper_params["tau"]
        bandit_data = "bandit_data_sampled_"
        path = f"../data/{dataset}_raw_linear/{tau_value}_0/"
        path += bandit_data + str(hyper_params["num_sample"])
        store_folder = "Test"
    else:
        store_folder = hyper_params.dataset["name"]
        path = f"../data/{store_folder}/bandit_data"
        path += "_sampled_" + str(hyper_params["num_sample"])
    
    print(store_folder)
    print(path)

    USE_NUE_VALUE = True
    
    train_path = path + "_train"
    val_path = path + "_val"
    test_path = path + "_test"
    if hyper_params["train_ratio"] is not None and create_dataset==False:
        train_path += "_" + str(hyper_params["train_ratio"])
        val_path += "_" + str(hyper_params["train_ratio"])
        test_path += "_" + str(hyper_params["train_ratio"])
    if hyper_params["unbalance"] is not None and create_dataset==False:
        bad_class_range = hyper_params["unbalance"][0]
        omit_prop = hyper_params["unbalance"][1]
        train_path += f"_(C={bad_class_range},P={omit_prop})"
        val_path += f"_(C={bad_class_range},P={omit_prop})"
        test_path += f"_(C={bad_class_range},P={omit_prop})"
    if hyper_params["gaussian_imbalance"] is not None and create_dataset==False:
        train_path += "_ImbalanceGaussian_" + str(hyper_params["gaussian_imbalance"])
        val_path += "_ImbalanceGaussian_" + str(hyper_params["gaussian_imbalance"])
        test_path += "_ImbalanceGaussian_" + str(hyper_params["gaussian_imbalance"])
    if hyper_params["data_repeat"] is not None and create_dataset==False:
        train_path += "_Rep=" + str(hyper_params["data_repeat"])
        val_path += "_Rep=" + str(hyper_params["data_repeat"])
        test_path += "_Rep=" + str(hyper_params["data_repeat"])
        
    if hyper_params["uniform_noise_alpha"] is not None and create_dataset==False:
        train_path += "_UniformNoise" + str(hyper_params["uniform_noise_alpha"])
        val_path += "_UniformNoise" + str(hyper_params["uniform_noise_alpha"])
        test_path += "_UniformNoise" + str(hyper_params["uniform_noise_alpha"])
    if hyper_params["gaussian_noise_alpha"] is not None and create_dataset==False:
        USE_NUE_VALUE = False
        train_path += "_GaussianNoise" + str(hyper_params["gaussian_noise_alpha"])
        val_path += "_GaussianNoise" + str(hyper_params["gaussian_noise_alpha"])
        test_path += "_GaussianNoise" + str(hyper_params["gaussian_noise_alpha"])
    if hyper_params["gamma_noise_beta"] is not None and create_dataset==False:
        USE_NUE_VALUE = False
        train_path += "_GammaNoise" + str(hyper_params["gamma_noise_beta"])
        val_path += "_GammaNoise" + str(hyper_params["gamma_noise_beta"])
        test_path += "_GammaNoise" + str(hyper_params["gamma_noise_beta"])  
    if hyper_params["biased_log_policy"] is not None and create_dataset==False:
        train_path += "_BiasedLoggingPolicy"
        val_path += "_BiasedLoggingPolicy"
        test_path += "_BiasedLoggingPolicy"
    if hyper_params["reward_flip"] is not None and create_dataset==False:
        train_path += "_RewardFlip" + str(hyper_params["reward_flip"])
        val_path += "_RewardFlip" + str(hyper_params["reward_flip"])
        test_path += "_RewardFlip" + str(hyper_params["reward_flip"])
    if hyper_params["logging_policy_cm"] is not None and create_dataset==False:
        splitted_cfm_path = hyper_params["logging_policy_cm"].split("/")
        train_path += f"CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
        val_path += f"CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"
        test_path += f"CFM_Policy_{splitted_cfm_path[-2]}_{splitted_cfm_path[-1]}"       
    if hyper_params["propensity_estimation"] is not None and create_dataset==False:
        train_path += "_propensity_estimation"
        val_path += "_propensity_estimation"
        test_path += "_propensity_estimation"
    
    print("Use nue value?", USE_NUE_VALUE)
    print("Train data path is", train_path)
    print("Validation data path is", val_path)
    print("Test data path is", test_path)
    
    labeled_train = None
    labeled_val = None
    if labeled:
        x_train, delta_train, prop_train, action_train = readfile(train_path, hyper_params)
        x_val, delta_val, prop_val, action_val = readfile(val_path, hyper_params)
    else:
        (
            x_train,
            delta_train,
            prop_train,
            action_train,
            labeled_train,
            flip_train,
        ) = readfile_unlabeled(train_path, hyper_params)
        x_val, delta_val, prop_val, action_val, labeled_val, flip_val = readfile_unlabeled(val_path, hyper_params)
    x_test, delta_test, prop_test, action_test = readfile(test_path, hyper_params)
    # Shuffle train set
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    delta_train = delta_train[indices]
    prop_train = prop_train[indices]
    action_train = action_train[indices]
    if hyper_params["reward_flip"] is not None:
        flip_train = flip_train[indices]
    if not labeled:
        labeled_train = labeled_train[indices]

    data = {}
    data_loader = {}
    if labeled:
        data["train"] = create_tensors(
            x_train,
            delta_train,
            prop_train,
            action_train,
            device="cpu",
            hyper_params=hyper_params,
            use_nue_value=USE_NUE_VALUE,
            flip=flip_train
        )
        data["val"] = create_tensors(
            x_val,
            delta_val,
            prop_val,
            action_val,
            device="cpu",
            hyper_params=hyper_params,
            use_nue_value=USE_NUE_VALUE,
            flip=flip_val
        )
    else:
        data["train"] = create_tensors(
            x_train,
            delta_train,
            prop_train,
            action_train,
            labeled=labeled_train,
            device="cpu",
            hyper_params=hyper_params,
            use_nue_value=USE_NUE_VALUE,
            flip=flip_train
        )
        data["val"] = create_tensors(
            x_val,
            delta_val,
            prop_val,
            action_val,
            labeled=labeled_val,
            device="cpu",
            hyper_params=hyper_params,
            use_nue_value=USE_NUE_VALUE,
            flip=flip_val
        )

    data["train"] = TensorDataset(*data["train"])
    data_loader["train"] = TorchDataLoader(
        data["train"], num_workers=0, batch_size=hyper_params["batch_size"]
    )

    data["val"] = TensorDataset(*data["val"])
    data_loader["val"] = TorchDataLoader(
        data["val"], num_workers=0, batch_size=hyper_params["batch_size"]
    )

    data["test"] = create_tensors(
        x_test,
        delta_test,
        prop_test,
        action_test,
        device="cpu",
        hyper_params=hyper_params,
        use_nue_value=USE_NUE_VALUE
    )
    data["test"] = TensorDataset(*data["test"])
    data_loader["test"] = TorchDataLoader(
        data["test"], num_workers=0, batch_size=hyper_params["batch_size"]
    )
    return data_loader["train"], data_loader["test"], data_loader["val"]
