import numpy as np
import pickle
from tqdm import tqdm

avg_num_zeros = 0.0
from code.model_h0 import ModelCifar
from functools import partial
from code.hyper_params import load_hyper_params
import argparse
import torch
import os
from code.utils import dataset_mapper
import timm
import torch.nn as nn

np.random.seed(2023)
torch.manual_seed(2023)


def one_hot(arr, num_classes):
    new = []
    for i in tqdm(range(len(arr))):
        temp = np.zeros(num_classes)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


# def get_probs(label):
#     probs = np.power(np.add(list(range(10)), 1), 0.5)
#     probs[label] = probs[label] * 10

#     probs = probs / probs.sum()

#     return probs


# def get_biased_probs(label):
#     probs = np.power(np.add(list(range(10)), 1), 0.5)
#     probs[label] = probs[label] * 1.5

#     probs = probs / probs.sum()

#     return probs

# def get_wrong_probs(label):
#     probs = np.ones(10)
#     probs[(label + 1) % 10] *= 10

#     probs = probs / probs.sum()

#     return probs


def get_uniform_probs(num_classes):
    probs = np.ones(num_classes)

    probs = probs / probs.sum()

    return probs


def get_feature_from_model(image, model, device):
    with torch.no_grad():
        # if dataset == "fmnist":
        #     image = torch.tensor(
        #         image.reshape(-1, 3, 28, 28).repeat(1, 3, 1, 1)
        #     ).float()
        # elif dataset == "cifar":
        #     image = torch.tensor(image.reshape(-1, 3, 32, 32)).float()
        # else:
        #     raise ValueError(f"Dataset {dataset} not valid.")
        out = (
            model(torch.tensor(image[None, :, :, :]).float().to(device))
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return out


# def get_model_probs_we(image, model, eps):
#     with torch.no_grad():
#         image = torch.tensor(
#             np.repeat(image.reshape(1, 1, 28, 28), repeats=3, axis=1)
#         ).float()
#         probs = (
#             torch.softmax(model(image.to(device)), dim=-1)
#             .cpu()
#             .numpy()
#             .squeeze(0)
#             .astype(np.float32)
#         )
#     probs[probs < eps] = eps
#     probs /= np.sum(probs)
#     return probs


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", required=True, help="Path to experiment config file."
)
parser.add_argument("-d", "--device", required=True, help="Device", type=str)
parser.add_argument("--tau", type=float, required=True, default=1.0)
parser.add_argument("--ul", type=float, required=True, default=0)
parser.add_argument("--reward_model", type=str, required=False)
parser.add_argument("--dataset", type=str, required=False)

args = parser.parse_args()
hyper_params = load_hyper_params(args.config)
ul_ratio = None
proportion = 1.0
labeled_proportion = 1.0
tau = args.tau
eps = 0

ul_ratio = args.ul
# dataset = 'biased'
dataset = None

if ul_ratio is not None:
    labeled_proportion = 1 / (ul_ratio + 1)

print("Labelled proportion =", labeled_proportion)

if ul_ratio is None:
    exit()
if proportion < 1.0:
    exit()
if dataset is not None:
    exit()
if eps > 0:
    exit()

dataset = args.dataset
hyper_params["dataset"] = dataset_mapper[dataset]
store_folder = dataset + "_pretrained_feature"

store_folder += "/"
# store_folder += f"{tau}"
# store_folder += f"_{int(ul_ratio)}"
store_folder += "base"
print(store_folder)
device = args.device

model = timm.create_model("resnet50", pretrained=True)
model.fc = nn.Identity()
model.to(device)
model.eval()

feature_fn = partial(get_feature_from_model, model=model, device=device)

os.makedirs(f"data/{store_folder}", exist_ok=True)


x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []


train_dataset = dataset_mapper[dataset]["class"](
    root=f"data/dataset/",
    train=True,
    **(dataset_mapper[dataset]["args"]),
    download=True,
)
test_dataset = dataset_mapper[dataset]["class"](
    root=f"data/dataset/",
    train=False,
    **(dataset_mapper[dataset]["args"]),
    download=True,
)
N, M = len(train_dataset), len(test_dataset)
print("Len Train =", N)
print("Len Test =", M)

# Train
for i in range(N):
    image, label = train_dataset[i]
    image = np.array(image).transpose(2, 0, 1).astype(np.float32)
    x_train.append(image)
    y_train.append(label)
x_train = np.stack(x_train)
y_train = np.stack(y_train)

# Test
for i in range(M):
    image, label = test_dataset[i]
    image = np.array(image).transpose(2, 0, 1).astype(np.float32)
    x_test.append(image)
    y_test.append(label)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

# Normalize X data
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0

# One hot the rewards
y_train = one_hot(y_train, dataset_mapper[dataset]["num_classes"])
y_test = one_hot(y_test, dataset_mapper[dataset]["num_classes"])

# Shuffle the dataset once
indices = np.arange(len(x_train))
np.random.shuffle(indices)
assert len(x_train) == len(y_train)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)
N = len(x_train)
n = int(N * labeled_proportion)
# Start creating bandit-dataset
print("x_train, x_val shape = ", x_train.shape, x_test.shape)
for num_sample in [1]:  # [1, 2, 3, 4, 5]:
    print("Pre-processing for num sample = " + str(num_sample))

    final_x, final_y, final_actions, final_prop, final_labeled = [], [], [], [], []

    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0
    neg_cost_count = 0

    for epoch in range(num_sample):
        for point_num in tqdm(range(x_train.shape[0])):
            image = x_train[point_num]
            label = np.argmax(y_train[point_num])

            probs = get_uniform_probs(
                num_classes=dataset_mapper[dataset]["num_classes"]
            )
            features = feature_fn(image)
            u = probs.astype(np.float64)
            actionvec = np.random.multinomial(1, u / np.sum(u))
            action = np.argmax(actionvec)

            final_x.append(features)
            final_actions.append([action])
            final_prop.append(probs)
            if labeled_proportion < 1.0:
                if point_num < n:
                    final_labeled.append(np.array([1.0]))
                else:
                    if label == action:
                        neg_cost_count += 1
                    final_labeled.append(np.array([0.0]))
            else:
                final_labeled.append(np.array([1.0]))

            final_y.append(y_train[point_num])

            expected_reward += float(int(action == label))
            total += 1.0
            # Printing the first prob. dist.
            # if point_num == 0: print("Prob Distr. for 0th sample:\n", [ round(i, 3) for i in list(probs) ])

    avg_num_zeros /= float(x_train.shape[0])
    avg_num_zeros = round(avg_num_zeros, 4)
    print(
        "Num sample = "
        + str(num_sample)
        + "; Acc = "
        + str(100.0 * expected_reward / total)
    )
    print("Neg reward proportion = " + str(neg_cost_count / total))
    print()

    # Save as CSV
    # if labeled_proportion < 1.0:
    final_normal = np.concatenate(
        (final_x, final_y, final_prop, final_actions, final_labeled), axis=1
    )
    print("final normal = ", final_normal.shape)
    # else:
    #     final_normal = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)

    N = len(final_normal)
    idx = list(range(N))
    idx = np.random.permutation(idx)
    print("Meta train size = ", dataset_mapper[dataset]["sizes"]["train"])
    train = final_normal[idx[: dataset_mapper[dataset]["sizes"]["train"]]]
    val = final_normal[idx[dataset_mapper[dataset]["sizes"]["train"] :]]
    print("train, val shape = ", train.shape, val.shape)
    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0

    test_prop, test_actions = [], []
    xs = []
    for i, label in tqdm(enumerate(y_test)):
        label = np.argmax(label)
        image = x_test[i]
        probs = get_uniform_probs(num_classes=dataset_mapper[dataset]["num_classes"])
        features = feature_fn(image)
        xs.append(features)
        u = probs.astype(np.float64)
        test_prop.append(probs)
        actionvec = np.random.multinomial(1, u / np.sum(u))
        action = np.argmax(actionvec)
        test_actions.append([action])

        expected_reward += float(int(action == label))
        total += 1.0

    print("Acc = " + str(100.0 * expected_reward / total))
    print()

    print("Hello " , xs[0].shape)
    # exit()

    test = np.concatenate((xs, y_test, test_prop, test_actions), axis=1)  # Garbage
    filename = f"data/{store_folder}/bandit_data_"
    filename += "sampled_" + str(num_sample)
    print("file name = ", filename)
    save_obj(train, filename + "_train")
    save_obj(test, filename + "_test")
    save_obj(val, filename + "_val")
