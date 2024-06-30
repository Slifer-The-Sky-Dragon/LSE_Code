import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import time
from tensorboardX import SummaryWriter

writer = None

from model import ModelCifar
from data import load_data
from eval import evaluate
from loss import CustomLoss, KLLoss, KLLossRev, SupKLLoss
from utils import *
from hyper_params import load_hyper_params
import argparse
import yaml
import numpy as np
import optuna
from functools import partial
import os
from copy import deepcopy
from optuna.trial import TrialState

def add_subpath(path, id):
    return path + "/" + str(id)
def dirname(path):
    return "/".join(path.split("/")[:-1])

def train(trial, model, criterion, optimizer, scheduler, reader, hyper_params, device):
    model.train()

    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([0.0])
    correct, total = LongTensor([0]), 0.0
    control_variate = FloatTensor([0.0])
    ips = FloatTensor([0.0])
    main_loss = FloatTensor([0.0])

    for x, y, action, delta, prop, labeled in reader.iter():
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()

        x, y, action, delta, prop = (
            x.to(device),
            y.to(device),
            action.to(device),
            delta.to(device),
            prop.to(device),
        )
        # Forward pass
        output = model(x)
        output = F.softmax(output, dim=1)
        output_labeled = output[labeled == 1]
        y_labeled = y[labeled == 1]
        delta_labeled = delta[labeled == 1]
        prop_labeled = prop[labeled == 1]
        action_labeled = action[labeled == 1]
        su = (labeled == 1).sum()
        if su > 0:
            if hyper_params.experiment.feedback == "supervised":
                loss = criterion(output_labeled, y_labeled)
            elif hyper_params.experiment.feedback == "bandit":
                loss = criterion(
                    output_labeled, action_labeled, delta_labeled, prop_labeled
                )
            elif hyper_params.experiment.feedback is None:
                loss = torch.tensor(0).float().to(x.device)
            else:
                raise ValueError(
                    f"Feedback type {hyper_params.experiment.feedback} is not valid."
                )
        else:
            loss = torch.tensor(0).float().to(x.device)
        main_loss += loss.item()
        if hyper_params.experiment.regularizers:
            if "KL" in hyper_params.experiment.regularizers:
                KL_coef = trial.suggest_float(
                    "KL_coef",
                    hyper_params.experiment.regularizers.KL[0],
                    hyper_params.experiment.regularizers.KL[1],
                    log=True,
                )
                loss += KLLoss(output, action, prop) * KL_coef
            if "KL2" in hyper_params.experiment.regularizers:
                KL2_coef = trial.suggest_float(
                    "KL2_coef",
                    hyper_params.experiment.regularizers.KL2[0],
                    hyper_params.experiment.regularizers.KL2[1],
                    log=True,
                )
                loss += KLLossRev(output, action, prop) * KL2_coef
            if "SupKL" in hyper_params.experiment.regularizers:
                if su > 0:
                    SupKL_coef = trial.suggest_float(
                        "SupKL_coef",
                        hyper_params.experiment.regularizers.SupKL[0],
                        hyper_params.experiment.regularizers.SupKL[1],
                        log=True,
                    )
                    loss += (
                        SupKLLoss(
                            output_labeled,
                            action_labeled,
                            delta_labeled,
                            prop_labeled,
                            hyper_params.experiment.regularizers.eps,
                        )
                        * SupKL_coef
                    )

        # print(loss.requires_grad)
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            if "lr_sch" in hyper_params and hyper_params["lr_sch"] == "OneCycle":
                scheduler.step()
        # else:
        #     print("No grad to optimize!!!")

        # Log to tensorboard
        writer.add_scalar("train loss", loss.item(), total_batches)

        # Metrics evaluation
        total_loss += loss.item()
        control_variate += torch.mean(
            output[range(action.size(0)), action] / prop
        ).item()
        ips += torch.mean((delta * output[range(action.size(0)), action]) / prop).item()
        predicted = torch.argmax(output, dim=1)
        # print(predicted, y)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        total_batches += 1.0
    if "lr_sch" not in hyper_params or hyper_params["lr_sch"] != "OneCycle":
        scheduler.step()

    metrics["main_loss"] = round(float(main_loss) / total_batches, 4)
    metrics["loss"] = round(float(total_loss) / total_batches, 4)
    metrics["Acc"] = round(100.0 * float(correct) / float(total), 4)
    metrics["CV"] = round(float(control_variate) / total_batches, 4)
    metrics["SNIPS"] = round(float(ips) / float(control_variate), 4)

    return metrics


def main(trial, hyper_params, device="cuda:0", return_model=False):
    # # If custom hyper_params are not passed, load from hyper_params.py
    # if hyper_params is None: from hyper_params import hyper_params
    hyper_params = deepcopy(hyper_params)
    hyper_params["tensorboard_path"] = add_subpath(
        hyper_params["tensorboard_path"], trial._trial_id
    )
    hyper_params["log_file"] = add_subpath(hyper_params["log_file"], trial._trial_id)
    hyper_params["summary_file"] = add_subpath(
        hyper_params["summary_file"], trial._trial_id
    )
    print(dirname(hyper_params["summary_file"]), hyper_params["summary_file"])
    os.makedirs(dirname(hyper_params["tensorboard_path"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["log_file"]), exist_ok=True)
    os.makedirs(dirname(hyper_params["summary_file"]), exist_ok=True)
    print(hyper_params)
    if hyper_params.experiment.regularizers:
        if "KL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer KL added: {hyper_params.experiment.regularizers.KL}"
            )
        if "KL2" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Reverse KL added: {hyper_params.experiment.regularizers.KL2}"
            )
        if "SupKL" in hyper_params.experiment.regularizers:
            print(
                f"--> Regularizer Supervised KL added: {hyper_params.experiment.regularizers.SupKL}"
            )

    # Initialize a tensorboard writer
    global writer
    path = hyper_params["tensorboard_path"]
    writer = SummaryWriter(path)

    # Train It..
    train_reader, test_reader, val_reader = load_data(hyper_params, labeled=False)

    file_write(
        hyper_params["log_file"],
        "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n",
    )
    file_write(hyper_params["log_file"], "Data reading complete!")
    file_write(
        hyper_params["log_file"],
        "Number of train batches: {:4d}".format(len(train_reader)),
    )
    file_write(
        hyper_params["log_file"],
        "Number of test batches: {:4d}".format(len(test_reader)),
    )

    if hyper_params.experiment.feedback == "supervised":
        print("Supervised Training.")
        criterion = nn.CrossEntropyLoss()
    elif hyper_params.experiment.feedback == "bandit":
        print("Bandit Training")
        criterion = CustomLoss(hyper_params)
    elif hyper_params.experiment.feedback is None:
        criterion = None
    else:
        raise ValueError(
            f"Feedback type {hyper_params.experiment.feedback} is not valid."
        )

    try:
        best_metrics_total = []
        weight_decay = trial.suggest_float(
            "weight_decay",
            hyper_params["weight_decay"][0],
            hyper_params["weight_decay"][1],
            log=True,
        )
        for exp in range(hyper_params.experiment.n_exp):
            model = ModelCifar(hyper_params).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=hyper_params["lr"],
                momentum=0.9,
                weight_decay=weight_decay,
            )
            if "lr_sch" in hyper_params and hyper_params["lr_sch"] == "OneCycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=hyper_params["lr"],
                    epochs=hyper_params["epochs"],
                    steps_per_epoch=len(train_reader),
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=20, gamma=0.5, verbose=True
                )
            file_write(
                hyper_params["log_file"], "\nModel Built!\nStarting Training...\n"
            )
            file_write(
                hyper_params["log_file"],
                f"################################ MODEL ITERATION {exp + 1}:\n--------------------------------",
            )
            best_acc = 0
            best_metrics = None
            for epoch in range(1, hyper_params["epochs"] + 1):
                epoch_start_time = time.time()

                # Training for one epoch
                metrics = train(
                    trial,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    train_reader,
                    hyper_params,
                    device,
                )

                string = ""
                for m in metrics:
                    string += " | " + m + " = " + str(metrics[m])
                string += " (TRAIN)"

                for metric in metrics:
                    writer.add_scalar(
                        f"Train_metrics/exp_{exp}/" + metric, metrics[metric], epoch - 1
                    )

                # Calulating the metrics on the validation set
                metrics = evaluate(
                    trial,
                    model,
                    criterion,
                    val_reader,
                    hyper_params,
                    device,
                    labeled=False,
                )
                string2 = ""
                for m in metrics:
                    string2 += " | " + m + " = " + str(metrics[m])
                string2 += " (VAL)"

                for metric in metrics:
                    writer.add_scalar(
                        f"Validation_metrics/exp_{exp}/" + metric,
                        metrics[metric],
                        epoch - 1,
                    )

                ss = "-" * 89
                ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                    epoch, (time.time() - epoch_start_time)
                )
                ss += string
                ss += "\n"
                ss += "-" * 89
                ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                    epoch, (time.time() - epoch_start_time)
                )
                ss += string2
                ss += "\n"
                ss += "-" * 89
                val_metrics = metrics
                
                if metrics["Acc"] > best_acc:
                    best_acc = metrics["Acc"]

                    metrics = evaluate(
                        trial,
                        model,
                        criterion,
                        test_reader,
                        hyper_params,
                        device,
                        labeled=True,
                    )
                    string3 = ""
                    for m in metrics:
                        string3 += " | " + m + " = " + str(metrics[m])
                    string3 += " (TEST)"

                    ss += "\n| end of epoch {:3d} | time: {:5.2f}s".format(
                        epoch, (time.time() - epoch_start_time)
                    )
                    ss += string3
                    ss += "\n"
                    ss += "-" * 89

                    for metric in metrics:
                        writer.add_scalar(
                            f"Test_metrics/exp_{exp}/" + metric,
                            metrics[metric],
                            epoch - 1,
                        )
                    best_metrics = metrics

                file_write(hyper_params["log_file"], ss)

                trial.report(val_metrics["Acc"], epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    best_metrics_total.append(best_metrics)
                    raise optuna.exceptions.TrialPruned()
                
            best_metrics_total.append(best_metrics)

    except KeyboardInterrupt:
        print("Exiting from training early")

    writer.close()

    model_summary = {k: [] for k in best_metrics_total[0].keys()}
    for metric in best_metrics_total:
        for k, v in metric.items():
            model_summary[k].append(v)
    for k, v in model_summary.items():
        model_summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    file_write(hyper_params["summary_file"], yaml.dump(model_summary))
    # trial.report(model_summary["Acc"]["mean"], )

    return model_summary["Acc"]["mean"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to experiment config file."
    )
    parser.add_argument("-d", "--device", required=True, help="Device", type=str)
    args = parser.parse_args()
    hyper_params = load_hyper_params(args.config)
    result_path = hyper_params["summary_file"] + "/final.txt"
    print(result_path)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(
            main, hyper_params=hyper_params, device=args.device, return_model=False
        ),
        n_trials=hyper_params.experiment.n_trials,
    )
    # best_metrics = main(args.config, device=args.device)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    result_path = hyper_params["log_file"] + "/final.txt"
    s = ""
    print("Study statistics: ")
    s += "Study statistics: \n"
    print("  Number of finished trials: ", len(study.trials))
    s +=  "  Number of finished trials: " + str(len(study.trials)) + "\n"
    print("  Number of pruned trials: ", len(pruned_trials))
    s += "  Number of pruned trials: " + str(len(pruned_trials)) + "\n"
    print("  Number of complete trials: ", len(complete_trials))
    s += "  Number of complete trials: " + str(len(complete_trials)) + "\n"

    print("Best trial:")
    s += "Best trial:\n"
    trial = study.best_trial

    print("  Value: ", trial.value)
    s += "  Value: " + str(trial.value) + "\n"
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        s += "    {}: {}".format(key, value) + "\n"
    file_write(result_path, s)