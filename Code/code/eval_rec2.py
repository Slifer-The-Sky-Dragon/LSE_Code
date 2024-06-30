import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from loss import KLLossRec, KLLossRevRec, AlphaRenyiLossRec
from tqdm import tqdm
from loss import KLLoss, KLLossRev, SupKLLoss
from utils import *


def evaluate(
    model, criterion, reader, hyper_params, device, labeled=True, as_reward=False
):
    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0]).to(device)
    main_loss = FloatTensor([0.0]).to(device)
    # correct, total = 0, 0.0
    # avg_correct = FloatTensor([0.0]).to(device)
    control_variate = FloatTensor([0.0]).to(device)
    ips = FloatTensor([0.0]).to(device)
    log_ips = FloatTensor([0.0]).to(device)
    tau = hyper_params["tau"] if "tau" in hyper_params else 1.0
    total = 0
    model.eval()

    for item in reader:
        x, action, prop, reward, _, items = item
        x, action, reward, prop, items = (
            x.to(device),
            action.to(device),
            reward.to(device),
            prop.to(device),
            items.to(device),
        )
        with torch.no_grad():
            output = model(x, items)
            output = F.softmax(output / tau, dim=1)

            if hyper_params.experiment.feedback == "supervised":
                if hyper_params["propensiy_estimation"] is not None:
                    loss = criterion(output, action)
                else:
                    loss = criterion(output, y)
            elif hyper_params.experiment.feedback == "bandit":
                if as_reward:
                    loss = criterion(output, action, reward)
                else:
                    loss = criterion(output, action, reward, prop)
            elif hyper_params.experiment.feedback is None:
                loss = torch.tensor(0).float().to(x.device)
            else:
                raise ValueError(
                    f"Feedback type {hyper_params.experiment.feedback} is not valid."
                )
            main_loss += loss
            total_loss += loss
            if hyper_params.experiment.regularizers:
                if "KL" in hyper_params.experiment.regularizers:
                    loss += (
                        KLLoss(
                            output,
                            action,
                            prop,
                            action_size=hyper_params["dataset"]["num_classes"],
                        )
                        * hyper_params.experiment.regularizers.KL
                    )
                if "KL2" in hyper_params.experiment.regularizers:
                    loss += (
                        KLLossRev(
                            output,
                            action,
                            prop,
                            action_size=hyper_params["dataset"]["num_classes"],
                        )
                        * hyper_params.experiment.regularizers.KL2
                    )
                if "SupKL" in hyper_params.experiment.regularizers:
                    loss += (
                        SupKLLoss(
                            output,
                            action,
                            reward,
                            prop,
                            hyper_params.experiment.regularizers.eps,
                            action_size=hyper_params["dataset"]["num_classes"],
                        )
                        * hyper_params.experiment.regularizers.SupKL
                    )

        control_variate += torch.sum(
            output[range(action.size(0)), action] / prop
        ).item()
        ips += torch.sum((reward * output[range(action.size(0)), action]) / prop).item()
        log_ips += torch.sum(reward).item()
        total += action.size(0)
        # predicted = torch.argmax(output, dim=1)
        # total += action.size(0)
        # print((predicted == y).sum().item())
        # if hyper_params["propensiy_estimation"] is not None:
        #     correct += (predicted == action).sum().item()
        # else:
        #     correct += (predicted == y).sum().item()
        # avg_correct += output[range(action.size(0)), y].sum().item()

        total_batches += 1.0
    # print("TOTAL EVAL BATCHES =", correct, total, total_batches)
    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    # metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    # metrics["AvgAcc"] = round(100.0 * float(avg_correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)
    metrics["IPS"] = round(float(ips) / total, 4)
    metrics["Log IPS"] = round(float(log_ips) / total, 4)

    return metrics
