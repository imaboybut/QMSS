
import importlib
import os
import sys
import time
from collections import OrderedDict
from math import sqrt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import wandb
import datetime

from torch.cuda.amp import autocast
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from config import datasetDict, trainTransformDict, testTransformDict
from metrics import metrics
from strategies import strategies as usual_strategies
from utilities.lr_schedulers import SequentialSchedulers, FixedLR
from utilities.utilities import Utilities as Utils
from utilities.utilities import WorstClassAccuracy, CalibrationError
from models.quantizer import QConv


class baseRunner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config):
        if not os.path.exists(config.log_dir):
            os.makedirs(os.path.join(config.log_dir, 'checkpoint'))


        self.config = config
        self.dataParallel = torch.cuda.device_count() > 1
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if "gpu" in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)

        # Set a couple useful variables
        self.checkpoint_file = None
        self.optimizer_file = None
        self.scheduler_file = None

        self.trained_test_accuracy = None
        self.trained_train_loss = None
        self.trained_train_accuracy = None
        self.after_pruning_metrics = None
        self.seed = None
        self.squared_model_norm = None
        self.n_warmup_epochs = None
        self.trainIterationCtr = 1
        self.tmp_dir = config["tmp_dir"]
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")
        self.ampGradScaler = (
            None  # Note: this must be reset before training, and before retraining
        )
        self.num_workers = None

        # Variables to be set by inheriting classes
        self.strategy = None
        self.ensemble_strategy = None
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
        self.trainLoader_unshuffled = None
        self.oodLoader = None
        self.n_datapoints = None
        self.model = None
        self.dense_model = None
        self.wd_scheduler = None
        self.trainData = None
        self.n_total_iterations = None
        # self.config.run_id = None
        self.boundaryRange = None
        self.ultimate_log_dict = None
        self.model_params_m =None
        self.model_params_q =None
        self.freeze_dict = None
        self.optimizer_m = None
        self.scheduler_m = None
        self.optimizer_q = None
        self.scheduler_q = None
        self.freeze_dict_file = None
        self.plot_dir = None
        self.base_plot_dir = "weight_distributions"
        self.trainLoader_ewgs = None


        if self.config.dataset in ["mnist", "cifar10"]:
            self.n_classes = 10
        elif self.config.dataset in ["cifar100"]:
            self.n_classes = 100
        elif self.config.dataset in ["tinyimagenet"]:
            self.n_classes = 200
        elif self.config.dataset in ["imagenet"]:
            self.n_classes = 1000
        else:
            raise NotImplementedError

        # Define the loss object and metrics
        # Important note: for the correct computation of loss/accuracy it's important to have reduction == 'mean'
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(
            device=self.device
        )

        self.metrics = {
            mode: {
                "loss": MeanMetric().to(device=self.device),
                "accuracy": Accuracy(num_classes=self.n_classes).to(device=self.device),
                "ips_throughput": MeanMetric().to(device=self.device),
            }
            for mode in ["train", "val", "test", "ood"]
        }
        for mode in ["val", "test", "ood"]:
            self.metrics[mode]["ece"] = CalibrationError(norm="l1").to(
                device=self.device
            )
            self.metrics[mode]["mce"] = CalibrationError(norm="max").to(
                device=self.device
            )
            self.metrics[mode]["worst_class_accuracy"] = WorstClassAccuracy(
                num_classes=self.n_classes
            ).to(device=self.device)

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self):
        with torch.no_grad():
            n_total, n_nonzero = metrics.get_parameter_count(model=self.model)

            x_input, y_target, indices = next(iter(self.valLoader))
            x_input, y_target = x_input.to(self.device), y_target.to(
                self.device
            )  # Move to CUDA if possible
            n_flops, n_nonzero_flops = metrics.get_flops(
                model=self.model, x_input=x_input
            )

            soup_metrics = (
                self.ensemble_strategy.get_ensemble_metrics()
                if self.ensemble_strategy is not None
                else {}
            )
            loggingDict = dict(
                train={
                    metric_name: metric.compute()
                    for metric_name, metric in self.metrics["train"].items()
                    if getattr(metric, "mode", True) is not None
                },  # Check if metric computable
                val={
                    metric_name: metric.compute()
                    for metric_name, metric in self.metrics["val"].items()
                },
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                nonzero_inference_flops=n_nonzero_flops,
                baseline_inference_flops=n_flops,
                theoretical_speedup=metrics.get_theoretical_speedup(
                    n_flops=n_flops, n_nonzero_flops=n_nonzero_flops
                ),
                learning_rate = (
                    float(self.optimizer.param_groups[0]["lr"])
                    if self.config.strategy == "Dense"
                    else float(self.optimizer_m.param_groups[0]["lr"])
                )
                ,
                distance_to_origin=metrics.get_distance_to_origin(self.model),
                soup_metrics=soup_metrics,
            )

            for split in ["test", "ood"]:
                loggingDict[split] = dict()
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[split][metric_name] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    def get_dataset_root(self, dataset_name: str) -> str:
        """Copies the dataset and returns the rootpath."""
        # Determine where the data lies
        for root in ["/data/"]:
            rootPath = f"{root}{dataset_name}"
            if os.path.isdir(rootPath):
                return rootPath
                break
        return f"{dataset_name}"

    def get_ood_dataloaders(self):
        if self.config.dataset == "cifar10":
            ood_dataset_name = "CIFAR10CORRUPT"
        elif self.config.dataset == "cifar100":
            ood_dataset_name = "CIFAR100CORRUPT"
        else:
            return None

        sys.stdout.write(f"Loading {ood_dataset_name} dataset for OOD performance.\n")
        ood_root = self.get_dataset_root(ood_dataset_name)
        ood_dataset = Utils.get_overloaded_dataset(datasetDict[ood_dataset_name])(
            root=ood_root, transform=testTransformDict[self.config.dataset]
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=self.num_workers,
        )

        return ood_loader

    def get_dataloaders(self):
        rootPath = self.get_dataset_root(dataset_name=self.config.dataset)
        print("root path : ", rootPath)

        if self.config.dataset in ["imagenet"]:
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=rootPath,
                split="train",
                transform=trainTransformDict[self.config.dataset],
            )
            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=rootPath,
                split="val",
                transform=testTransformDict[self.config.dataset],
            )
        elif self.config.dataset == "tinyimagenet":
            traindir = os.path.join(rootPath, "train_preprocess")
            valdir = os.path.join(rootPath, "valid_preprocess")
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=traindir, transform=trainTransformDict[self.config.dataset]
            )
            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=valdir, transform=testTransformDict[self.config.dataset]
            )
        else:
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=rootPath,
                train=True,
                download=True,
                transform=trainTransformDict[self.config.dataset],
            )

            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(
                root=rootPath,
                train=False,
                transform=testTransformDict[self.config.dataset],
            )
        train_size = int(0.9 * len(trainData))
        val_size = len(trainData) - train_size
        self.trainData, valData = torch.utils.data.random_split(
            trainData,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.n_datapoints = train_size

        if self.config.dataset in ["imagenet", "cifar100", "tinyimagenet"]:
            self.num_workers = (
                4 * torch.cuda.device_count() if torch.cuda.is_available() else 0
            )
        else:
            self.num_workers = 2 if torch.cuda.is_available() else 0
            
        self.num_workers = 2
        trainLoader = torch.utils.data.DataLoader(
            self.trainData,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=self.num_workers,
        )
        trainLoader_unshuffled = torch.utils.data.DataLoader(
            self.trainData,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )
        valLoader = torch.utils.data.DataLoader(
            valData,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )
        testLoader = torch.utils.data.DataLoader(
            testData,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )
        self.trainLoader_ewgs = torch.utils.data.DataLoader(
            self.trainData,
            batch_size=32,
            shuffle=True,
            pin_memory=False,
            num_workers=self.num_workers,
        )

        return trainLoader, valLoader, testLoader, trainLoader_unshuffled

    def load_optimizer(self, temporary: bool = True):
        
        
        file = self.optimizer_file
        assert file is not None

        dir = wandb.run.dir if not temporary else self.tmp_dir
        fPath = os.path.join(dir, file)

        parent_dir = os.path.dirname(fPath)

        if self.config.strategy == "Dense":
            state_dict = torch.load(fPath, map_location=self.device)
            self.optimizer.load_state_dict(state_dict)
        else:
            optimizer_state_dict_all = torch.load(fPath , map_location=self.device)
            optimizer_state_dict_m = optimizer_state_dict_all["optimizer_m"]
            optimizer_state_dict_q = optimizer_state_dict_all["optimizer_q"]
            


            self.optimizer_m.load_state_dict(optimizer_state_dict_m)
            self.optimizer_q.load_state_dict(optimizer_state_dict_q)
            
            lrs_m = [g['lr'] for g in self.optimizer_m.param_groups]
            print(f"[m] optimizer_m lr:", lrs_m)


            

        
        

    def load_scheduler(self, temporary: bool = True):
        file = self.scheduler_file
        assert file is not None

        dir = wandb.run.dir if not temporary else self.tmp_dir
        print(dir)
        fPath = os.path.join(dir, file)
        if self.config.strategy == "Dense":
            state_dict = torch.load(fPath, map_location=self.device)
            self.scheduler.load_state_dict(state_dict)
        else: 
            scheduler_state_dict_all = torch.load(fPath , map_location=self.device)
            scheduler_state_dict_m = scheduler_state_dict_all["scheduler_m"]
            scheduler_state_dict_q = scheduler_state_dict_all["scheduler_q"]

            self.scheduler_m.load_state_dict(scheduler_state_dict_m)
            self.scheduler_q.load_state_dict(scheduler_state_dict_q)
            


    def get_model(self, reinit: bool, temporary: bool = True) -> torch.nn.Module:

        model = getattr(
            importlib.import_module("models." + self.config.dataset),
            self.config.arch,)(self.config)

        file = self.checkpoint_file

        if file is not None:

            dir = wandb.run.dir if not temporary else self.tmp_dir
            fPath = os.path.join(dir, file)
            state_dict = torch.load(fPath, map_location=self.device)
            new_state_dict = OrderedDict()
            require_DP_format = isinstance(
                model, torch.nn.DataParallel
            )  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                new_state_dict[name] = v

            if self.config.strategy == "dense":
                print("dense model load")
                model.load_state_dict(new_state_dict)

            elif self.config.strategy == "WarmupQAT":
                print("getmodel_load WarmupQAT")
                # Load dense (FP32) model
                current_dict = model.state_dict()
                matched_layers = []
                for key in new_state_dict.keys():
                    if key in current_dict.keys():
                        current_dict[key].copy_(new_state_dict[key])
                        matched_layers.append(key)

                for layer in matched_layers:
                    print(f"   - {layer}")
                model.load_state_dict(current_dict, strict=False)


                self.init_quant_model(model, self.trainLoader, self.device)
                trainable_params = list(model.parameters())
                model_params = []
                quant_params = []
                for m in model.modules():
                    if isinstance(m, QConv):
                        model_params.append(m.weight)
                        if m.bias is not None:
                            model_params.append(m.bias)
                        if m.quan_weight:
                            quant_params.append(m.lW)
                            quant_params.append(m.uW)
                        if m.quan_act:
                            quant_params.append(m.lA)
                            quant_params.append(m.uA)
                        if m.quan_act or m.quan_weight:
                            quant_params.append(m.output_scale)
                    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        model_params.append(m.weight)
                        if m.bias is not None:
                            model_params.append(m.bias)
                    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                        if m.affine:
                            model_params.append(m.weight)
                            model_params.append(m.bias)
                print("# total params:", sum(p.numel() for p in trainable_params))
                print("# model params:", sum(p.numel() for p in model_params))
                print("# quantizer params:", sum(p.numel() for p in quant_params))
                if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
                    raise Exception('Mismatched number of trainable parmas')
                    
                iterations_per_epoch = len(self.trainLoader) 
                n_total_iterations = (
                    self.config.total_epochs
                )  

                self.optimizer_m = torch.optim.Adam(model_params, lr=self.config.lr_m, weight_decay=self.config.weight_decay)
                self.optimizer_q = torch.optim.Adam(quant_params, lr=self.config.lr_q)
                self.scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_m, T_max=n_total_iterations, eta_min=0.0)
                self.scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q, T_max=n_total_iterations, eta_min=0.0) 
            
                
            else:
                # Load LSQ Quantizated model
                print("getmodel_load quantization")

                self.init_quant_model(model, self.trainLoader, self.device)
                trainable_params = list(model.parameters())
                model_params = []
                quant_params = []
                for m in model.modules():
                    if isinstance(m, QConv):
                        model_params.append(m.weight)
                        if m.bias is not None:
                            model_params.append(m.bias)
                        if m.quan_weight:
                            quant_params.append(m.lW)
                            quant_params.append(m.uW)
                        if m.quan_act:
                            quant_params.append(m.lA)
                            quant_params.append(m.uA)
                        if m.quan_act or m.quan_weight:
                            quant_params.append(m.output_scale)
                    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        model_params.append(m.weight)
                        if m.bias is not None:
                            model_params.append(m.bias)
                    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                        if m.affine:
                            model_params.append(m.weight)
                            model_params.append(m.bias)
                print("# total params:", sum(p.numel() for p in trainable_params))
                print("# model params:", sum(p.numel() for p in model_params))
                print("# quantizer params:", sum(p.numel() for p in quant_params))
                if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
                    raise Exception('Mismatched number of trainable parmas')
                    
                iterations_per_epoch = len(self.trainLoader) 
                n_total_iterations = (
                    self.config.total_epochs
                )  
                self.model_params_m  = model_params
                self.model_params_q = quant_params
                self.optimizer_m = torch.optim.Adam(model_params, lr=self.config.lr_m, weight_decay=self.config.weight_decay)
                self.optimizer_q = torch.optim.Adam(quant_params, lr=self.config.lr_q)
                self.scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_m, T_max=n_total_iterations, eta_min=0.0)
                self.scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q, T_max=n_total_iterations, eta_min=0.0)
            
                self.load_optimizer()
                self.load_scheduler()
                
                    
                    
                model.load_state_dict(new_state_dict)
                for name, m in model.named_modules():
                    if name == "layer2.1.conv1" and isinstance(m, QConv):
                        if isinstance(m, QConv):
                            # get the tensor on CPU for printing
                            lW = m.lW.detach().cpu()
                            uW = m.uW.detach().cpu()
                            print(name, "lW =", m.lW)
                            print(name, "uW =", m.uW)
                

        if (
            self.dataParallel
            and reinit
            and not isinstance(model, torch.nn.DataParallel)
        ):  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)

        return model


    def define_optimizer_scheduler(self):
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = self.config.learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == "(":
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ")":
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(",")
        if name in [
            "ContinueCosine",
        ]:
            scheduler = (name, kwargs)
            self.initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        if self.config.optimizer == "SGD":
            wd = self.config["weight_decay"] or 0.0
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.initial_lr,
                momentum=self.config.momentum,
                weight_decay=wd,
                nesterov=wd > 0.0,
            )

            
        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = len(self.trainLoader)
        name, kwargs = scheduler
        print("Using {} learning rate scheduler".format(name))

        print("schedule epoch : ", self.config.total_epochs)
        n_total_iterations = (
            iterations_per_epoch * self.config.total_epochs
        )  

        self.n_total_iterations = n_total_iterations

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_total_iterations, eta_min=0.0
        )

        self.scheduler = scheduler

    def define_strategy(self, use_dense_base=False):
        #### UNSTRUCTURED
        # Define callbacksfinetuning_callback, restore_callback, save_model_callback
        callbackDict = {
            "after_pruning_callback": self.after_pruning_callback,
            "finetuning_callback": self.fine_tuning,
            "restore_callback": self.restore_model,
            "save_model_callback": self.save_model,
            "final_log_callback": self.final_log,
        }
        # Base strategies
        if use_dense_base or self.config.strategy == "WarmupQAT":
            return getattr(usual_strategies, "Dense")(
                model=self.model,
                n_classes=self.n_classes,
                config=self.config,
                callbacks=callbackDict,
            )
        else:
            print("Not use_dense_base", self.config.strategy)
            return getattr(usual_strategies, self.config.strategy)(
                model=self.model,
                n_classes=self.n_classes,
                config=self.config,
                callbacks=callbackDict,
            )

    def log(self, runTime, finetuning: bool = False, final_logging: bool = False):
        loggingDict = self.get_metrics()
        loggingDict.update({"epoch_run_time": runTime})
        if not finetuning:
            # Update final trained metrics (necessary to be able to filter via wandb)
            for metric_type, val in loggingDict.items():
                wandb.run.summary[f"final.{metric_type}"] = val
            # The usual logging of one epoch
            wandb.log(loggingDict)

        else:
            if not final_logging:
                wandb.log(
                    dict(
                        finetune=loggingDict,
                    ),
                )
            else:
                # We add the after_pruning_metrics and don't commit, since the values are updated by self.final_log
                self.ultimate_log_dict = dict(
                    finetune=loggingDict,
                    pruned=self.after_pruning_metrics,
                )

    def final_log(self):
        """This function can ONLY be called by pretrained strategies using the final sparsified model"""
        # Recompute accuracy and loss
        sys.stdout.write(f"\nFinal logging\n")
        self.reset_averaged_metrics()
        if self.config.strategy != "Dense":
            # We recalibrate the BN statistics also for IMP
            self.recalibrate_bn()
        if self.config.strategy == "Dense":
            self.evaluate_model(data="val")
            self.evaluate_model(data="test")
        else:
            
            final_test_acc = self.test_ewgs()
        
        # self.evaluate_model(data="ood")

        # Update final trained metrics (necessary to be able to filter via wandb)
        loggingDict = self.get_metrics()
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"final.{metric_type}"] = val

        # Update after prune metrics
        if self.after_pruning_metrics is not None:
            for metric_type, val in self.after_pruning_metrics.items():
                wandb.run.summary[f"pruned.{metric_type}"] = val

        # Add to existing self.ultimate_log_dict which was not committed yet
        if self.ultimate_log_dict is not None:
            if loggingDict["train"]["accuracy"] == 0:
                # we did not perform the recomputation, use the old values for train
                del loggingDict["train"]

            self.ultimate_log_dict["finetune"].update(loggingDict)
        else:
            self.ultimate_log_dict = {"finetune": loggingDict}
        print(f"WandB Summary Keys After Logging: {list(wandb.summary.keys())}")
        wandb.log(self.ultimate_log_dict)
        wandb.summary["real_my_test_acc_"] = final_test_acc

    def after_pruning_callback(self):
        """Collects pruning metrics. Is called ONCE per run, namely on the LAST PRUNING step."""

        self.reset_averaged_metrics()
        self.test_ewgs()

        norm_drop, relative_norm_drop = {}, {}

        self.after_pruning_metrics = dict(
            val={
                metric_name: metric.compute()
                for metric_name, metric in self.metrics["val"].items()
            },
            test={
                metric_name: metric.compute()
                for metric_name, metric in self.metrics["test"].items()
            },
            norm_drop=norm_drop,
            relative_norm_drop=relative_norm_drop,
        )
        
        # Reset squared model norm for following pruning steps, otherwise ALLR does not work properly
        # self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

    def restore_model(self) -> None:
        sys.stdout.write(f"Restoring model from {self.checkpoint_file}.\n")
        self.model = self.get_model(reinit=False, temporary=True)

    def save_model(
        self,
        model_type: str,
        temporary: bool = False,
    ) -> str:
        print("start save_model")
        if model_type not in ["initial", "trained", "warmup_qat", "qat", "ensemble"]:
            print(f"Ignoring to save {model_type} for now.")
            return None

        fName = f"{model_type}_model.pt"
        if model_type == "qat":
            fName = f"{model_type}_model_{self.config.split_val}_{self.config.phase}.pt"
        elif model_type == "ensemble":
            fName = f"{model_type}_model_{self.config.ensemble_method}_{self.config.phase}.pt"

        # Determine save directory

        save_dir = wandb.run.dir if not temporary else self.tmp_dir
        fPath = os.path.join(save_dir, fName)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        print("fPath : ", fPath)
        # Only save models in their non-module version, to avoid problems when loading
        
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        # Save the model state dict
        torch.save(model_state_dict, fPath)

        wandb.save(fPath)
        print("saved")

        return fPath

    def save_optimizer(
        self,
        model_type: str,
        temporary: bool = False,
    ) -> str:
        print("start save_optimizer")
        if model_type not in ["initial", "trained", "warmup_qat", "qat", "ensemble"]:
            print(f"Ignoring to save {model_type} for now.")
            return None
        fName = f"{model_type}_optimizer.pt"
        if model_type == "qat":
            fName = (
                f"{model_type}_optimizer_{self.config.split_val}_{self.config.phase}.pt"
            )
        elif model_type == "ensemble":
            fName = f"{model_type}_optimizer_{self.config.ensemble_method}_{self.config.phase}.pt"
        fPath = (
            os.path.join(wandb.run.dir, fName)
            if not temporary
            else os.path.join(self.tmp_dir, fName)
        )

        print("fPath : ", fPath)
        # Only save models in their non-module version, to avoid problems when loading
        if self.config.strategy == "Dense":
            try:
                optimizer_state_dict = self.optimizer.module.state_dict()
            except AttributeError:
                optimizer_state_dict = self.optimizer.state_dict()

            torch.save(optimizer_state_dict, fPath)  # Save the state_dict
            print("saved")
            return fPath
        else:
            try:
                optimizer_state_dict_m = self.optimizer_m.module.state_dict()
                optimizer_state_dict_q = self.optimizer_q.module.state_dict()
            except AttributeError:
                optimizer_state_dict_m = self.optimizer_m.state_dict()
                optimizer_state_dict_q = self.optimizer_q.state_dict()
                
            optimizer_state_dict_all = {
                "optimizer_m": optimizer_state_dict_m,
                "optimizer_q": optimizer_state_dict_q
            }

            torch.save(optimizer_state_dict_all, fPath)  # Save both state_dicts in one file
            print("optimizer_state_dict_all saved")
            return fPath
        

    def save_scheduler(
        self,
        model_type: str,
        temporary: bool = False,
    ) -> str:
        print("start save_scheduler")
        if model_type not in ["initial", "trained", "warmup_qat", "qat", "ensemble"]:
            print(f"Ignoring to save {model_type} for now.")
            return None
        fName = f"{model_type}_scheduler.pt"
        if model_type == "qat":
            fName = (
                f"{model_type}_scheduler_{self.config.split_val}_{self.config.phase}.pt"
            )
        elif model_type == "ensemble":
            fName = f"{model_type}_scheduler_{self.config.ensemble_method}_{self.config.phase}.pt"
        fPath = (
            os.path.join(wandb.run.dir, fName)
            if not temporary
            else os.path.join(self.tmp_dir, fName)
        )

        print("fPath : ", fPath)
        # Only save models in their non-module version, to avoid problems when loading
        if self.config.strategy == "Dense":
            try:
                scheduler_state_dict = self.scheduler.module.state_dict()
            except AttributeError:
                scheduler_state_dict = self.scheduler.state_dict()

            torch.save(scheduler_state_dict, fPath)  # Save the state_dict
            print("saved")
            return fPath
        
        
        else:
            try:
                scheduler_state_dict_m = self.scheduler_m.module.state_dict()
                scheduler_state_dict_q = self.scheduler_q.module.state_dict()
            except AttributeError:
                scheduler_state_dict_m = self.scheduler_m.state_dict()
                scheduler_state_dict_q = self.scheduler_q.state_dict()
                
            scheduler_state_dict_all = {
                "scheduler_m": scheduler_state_dict_m,
                "scheduler_q": scheduler_state_dict_q
            }

            torch.save(scheduler_state_dict_all, fPath)  # Save both state_dicts in one file
            print("scheduler_state_dict_all saved")
            return fPath
        

    def evaluate_model(self, data="train"):
        metrics = self.train_epoch(data=data, is_training=False)
        print(f"{data} Accuracy: {metrics['accuracy']}, Loss: {metrics['loss']}")
        return metrics


    def get_model_imagenet(self,model_name):
        model_class = globals().get(model_name)
        model = model_class(self.config, pretrained=True)
        self.model = model  
        self.init_quant_model(model, self.trainLoader, self.device) 
        trainable_params = list(model.parameters())
        model_params = []
        quant_params = []
        for m in model.modules():
            if isinstance(m, QConv):
                model_params.append(m.weight)
                if m.bias is not None:
                    model_params.append(m.bias)
                if m.quan_weight:
                    quant_params.append(m.lW)
                    quant_params.append(m.uW)
                if m.quan_act:
                    quant_params.append(m.lA)
                    quant_params.append(m.uA)
                if m.quan_act or m.quan_weight:
                    quant_params.append(m.output_scale)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                model_params.append(m.weight)
                if m.bias is not None:
                    model_params.append(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    model_params.append(m.weight)
                    model_params.append(m.bias)
        print("# total params:", sum(p.numel() for p in trainable_params))
        print("# model params:", sum(p.numel() for p in model_params))
        print("# quantizer params:", sum(p.numel() for p in quant_params))
        if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
            raise Exception('Mismatched number of trainable parmas')
            
        iterations_per_epoch = len(self.trainLoader) 
        n_total_iterations = (
            self.config.total_epochs
        )  
        print("self.config.lr_m : " , self.config.lr_m)          
        self.optimizer_m = torch.optim.Adam(model_params, lr=self.config.lr_m, momentum=self.config.momentum,weight_decay=self.config.weight_decay)
        print("model lr : " ,self.optimizer_m.param_groups[0]['lr'])
        self.optimizer_q = torch.optim.Adam(quant_params, lr=self.config.lr_q)
        self.scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_m, T_max=n_total_iterations, eta_min=0.0)
        self.model_params_m  = model_params
        self.scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q, T_max=n_total_iterations, eta_min=0.0) 

        return model


    def get_model_after(self,model_name , temporary: bool = True ,reinit: bool = True ) :

        model_class = globals().get(model_name)
        model = model_class(self.config, pretrained=True)
        file = self.checkpoint_file
        if file is not None:
            dir = wandb.run.dir if not temporary else self.tmp_dir
            fPath = os.path.join(dir, file)

            state_dict = torch.load(fPath, map_location=self.device)
            new_state_dict = OrderedDict()

            require_DP_format = isinstance(
                model, torch.nn.DataParallel
            )  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                new_state_dict[name] = v

            model_class = globals().get(model_name)
            model = model_class(self.config, pretrained=True)
            self.model = model  
            self.init_quant_model(model, self.trainLoader, self.device) 
            trainable_params = list(model.parameters())
            model_params = []
            quant_params = []
            for m in model.modules():
                if isinstance(m, QConv):
                    model_params.append(m.weight)
                    if m.bias is not None:
                        model_params.append(m.bias)
                    if m.quan_weight:
                        quant_params.append(m.lW)
                        quant_params.append(m.uW)
                    if m.quan_act:
                        quant_params.append(m.lA)
                        quant_params.append(m.uA)
                    if m.quan_act or m.quan_weight:
                        quant_params.append(m.output_scale)
                elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    model_params.append(m.weight)
                    if m.bias is not None:
                        model_params.append(m.bias)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    if m.affine:
                        model_params.append(m.weight)
                        model_params.append(m.bias)
            print("# total params:", sum(p.numel() for p in trainable_params))
            print("# model params:", sum(p.numel() for p in model_params))
            print("# quantizer params:", sum(p.numel() for p in quant_params))
            if sum(p.numel() for p in trainable_params) != sum(p.numel() for p in model_params) + sum(p.numel() for p in quant_params):
                raise Exception('Mismatched number of trainable parmas')
                
            iterations_per_epoch = len(self.trainLoader) 
            n_total_iterations = (
                self.config.total_epochs
            )  
            self.model_params_m = model_params
            self.model_params_q = quant_params         
            self.optimizer_m = torch.optim.Adam(model_params, lr=self.config.lr_m, momentum=self.config.momentum,weight_decay=self.config.weight_decay)
            self.optimizer_q = torch.optim.Adam(quant_params, lr=self.config.lr_q)
            self.scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_m, T_max=n_total_iterations, eta_min=0.0)
            self.scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q, T_max=n_total_iterations, eta_min=0.0) 
            self.load_optimizer()
            self.load_scheduler()

            model.load_state_dict(new_state_dict)


        if (
            self.dataParallel
            and reinit
            and not isinstance(model, torch.nn.DataParallel)
        ):  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)

        return model
    
    def fine_tuning(self, n_epochs_finetune, phase=1):
        phase = self.config.phase
        print("n_epochs_finetune : ", n_epochs_finetune)
        if n_epochs_finetune == 0:
            return
        if self.config.ensemble_by == "retrain_length":
            n_epochs_finetune = self.config.split_val
            sys.stdout.write(
                f"We split by the retrain length. Value {n_epochs_finetune}.\n"
            )
        n_phases = self.config.n_phases

        # Reset the GradScaler for AutoCast
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=False)


        self.strategy.set_to_finetuning_phase() 
        for epoch in range(1, n_epochs_finetune + 1, 1): 
            self.reset_averaged_metrics()
            
            sys.stdout.write(
                f"\nFinetuning: phase {phase}/{n_phases} | epoch {epoch}/{n_epochs_finetune}\n"
            )
            # Train
            t = time.time()
            if self.config.strategy =="Dense":
                self.train_epoch(data="train")
                self.evaluate_model(data="val")
            else:
                print("before finetune")
                self.test_ewgs()
                self.train_ewgs(epoch=n_epochs_finetune)
                # self.strategy.at_epoch_end(epoch=epoch)
                print("afterfintune")
                self.test_ewgs()
                self.log(
                    runTime=time.time() - t,
                    finetuning=True,
                    final_logging=(epoch == n_epochs_finetune and phase == n_phases),
                )
                break


            self.strategy.at_epoch_end(epoch=epoch)
            self.log(
                runTime=time.time() - t,
                finetuning=True,
                final_logging=(epoch == n_epochs_finetune and phase == n_phases),
            )

    def train_epoch(self, data="train", is_training=True):
        assert not (
            data in ["test", "val", "ood"] and is_training
        ), "Can't train on test/val/ood set."
        loaderDict = {
            "train": self.trainLoader,
            "val": self.valLoader,
            "test": self.testLoader,
            "ood": self.oodLoader,
        }
        loader = loaderDict[data]
        if loader is None and data == "ood":
            sys.stdout.write(f"No OOD data available. Skipping.\n")
            return

        (
            sys.stdout.write(f"Training:\n")
            if is_training
            else sys.stdout.write(f"Evaluation of {data} data:\n")
        )

        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, y_target, indices in pbar:
                    # Move to CUDA if possible
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()  # Zero the gradient buffers

                    itStartTime = time.time()

                    with autocast(enabled=(self.config.use_amp is True)):
                        output = self.model.train(mode=(data == "train"))(x_input)
                        loss = self.loss_criterion(output, y_target)

                    if is_training:
                        self.ampGradScaler.scale(
                            loss
                        ).backward()  # Scaling + Backpropagation
                        self.ampGradScaler.step(self.optimizer)  # Optimization step
                        self.ampGradScaler.update()

                        self.strategy.after_training_iteration(
                            it=self.trainIterationCtr,
                            lr=float(self.optimizer.param_groups[0]["lr"]),
                        )
                        self.scheduler.step()
                        self.trainIterationCtr += 1

                    itEndTime = time.time()
                    n_img_in_iteration = len(y_target)
                    ips = n_img_in_iteration / (
                        itEndTime - itStartTime
                    )  # Images processed per second

                    self.metrics[data]["loss"](value=loss, weight=len(y_target))
                    self.metrics[data]["accuracy"](output, y_target)
                    self.metrics[data]["ips_throughput"](ips)
                    if data in ["val", "test"]:
                        self.metrics[data]["ece"](output, y_target)
                        self.metrics[data]["mce"](output, y_target)
                        self.metrics[data]["worst_class_accuracy"](output, y_target)
                        
        final_accuracy = self.metrics[data]["accuracy"].compute().item()
        final_loss = self.metrics[data]["loss"].compute().item()
        
        return {'accuracy': final_accuracy, 'loss': final_loss}

    def train_ewgs(self,epoch):
        device = self.device
        total_iter = 0
        best_acc = 0
        
        for ep in range(epoch):
            # if self.config.phase == 1:
            #     self.model.train()
            # else:
            #     self.model.eval()
            self.model.train()
            ### update grad scales
            if ep % 10 == 0 and ep != 0:
                self.update_grad_scales(self.model, self.trainLoader_ewgs, self.loss_criterion, self.device, self.config)

            ###
            print('train/model_lr', self.optimizer_m.param_groups[0]['lr'], ep)
            print('train/quant_lr', self.optimizer_q.param_groups[0]['lr'], ep)
            for i, (images, labels , _) in enumerate(self.trainLoader):
                images = images.to(device)
                labels = labels.to(device)

                self.optimizer_m.zero_grad()
                self.optimizer_q.zero_grad()

                pred = self.model(images)
                loss_t = self.loss_criterion(pred, labels)

                loss = loss_t
                loss.backward()

                self.optimizer_m.step()
                if self.config.update_quant =="True":
                    self.optimizer_q.step()
                total_iter += 1

            self.scheduler_m.step()
            print(f"self.update_quant == {self.config.update_quant} ")
            if self.config.update_quant =="True":
                self.scheduler_q.step()


            
            with torch.no_grad():
                self.model.eval()
                correct_classified = 0
                total = 0
                for i, (images, labels, _) in enumerate(self.trainLoader):
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = self.model(images)
                    _, predicted = torch.max(pred.data, 1)
                    total += pred.size(0)
                    correct_classified += (predicted == labels).sum().item()
                test_acc = correct_classified/total*100
                print('train/acc', test_acc, ep)

                self.model.eval()
                correct_classified = 0
                total = 0
                for i, (images, labels,_) in enumerate(self.testLoader):
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = self.model(images)
                    _, predicted = torch.max(pred.data, 1)
                    total += pred.size(0)
                    correct_classified += (predicted == labels).sum().item()
                test_acc = correct_classified/total*100
                print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
                print('test/acc', test_acc, ep)


            
            layer_num = 0
            for m in self.model.modules():
                if isinstance(m, QConv):
                    layer_num += 1

                    
    def test_ewgs(self):
        device = self.device
        self.model.eval()
        with torch.no_grad():
            correct_classified = 0
            total = 0
            for i, (images, labels,_) in enumerate(self.testLoader):
                images = images.to(device)
                labels = labels.to(device)
                pred = self.model(images)
                _, predicted = torch.max(pred.data, 1)
                total += pred.size(0)
                correct_classified += (predicted == labels).sum().item()
            test_acc = correct_classified/total*100
            print("Test accuracy: {}%".format(test_acc))
            return test_acc
        
        
            
            
    def train(self):
        self.ampGradScaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.use_amp is True)
        )
        for epoch in range(self.config.n_epochs + 1):
            if self.config.strategy != "Dense": 
                pass
                
            self.reset_averaged_metrics()
            sys.stdout.write(f"\n\nEpoch {epoch}/{self.config.n_epochs}\n")
            t = time.time()
            if epoch > 0:
                # Train
                if self.config.strategy == "Dense":
                    self.train_epoch(data="train")
                else:
                    self.test_ewgs()
                    self.train_ewgs(epoch = self.config.n_epochs)
                    self.strategy.at_epoch_end(epoch=epoch)
                    self.test_ewgs()
                    self.recalibrate_bn()
                    self.test_ewgs()
                    self.log(runTime=time.time() - t)
                    break 
                
            if epoch == self.config.n_epochs:
                # Do one complete evaluation on the test data set
                
                if self.config.strategy == "Dense":
                    self.recalibrate_bn()
                    self.evaluate_model(data="test")
                else:
                    #1. test_ewgs , 2. test 3. bn 4. test
                    self.test_ewgs()
                    # self.evaluate_model(data="test")
                    self.recalibrate_bn()
                    self.test_ewgs()
                    # self.evaluate_model(data="test")
            self.strategy.at_epoch_end(epoch=epoch)

            self.log(runTime=time.time() - t)
            

        self.trained_test_accuracy = self.metrics["test"]["accuracy"].compute()
        self.trained_train_loss = self.metrics["train"]["loss"].compute()

    def recalibrate_bn(self):
        # Reset BN statistics
        self.model.train()
        recalibration_fraction = self.config.bn_recalibration_frac #
        if self.config.bn_recalibration_frac is None or not (
            0 <= self.config.bn_recalibration_frac <= 1
        ):
            recalibration_fraction = 1.0
            sys.stdout.write(
                f"bn_recalibration_frac not specified or invalid ({self.config.bn_recalibration_frac}). Recalibrating BN-statistics on 100% of the training data (unshuffled).\n"
            )

        reset_ctr = 0
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.reset_running_stats()
                reset_ctr +=1
        sys.stdout.write(
            f"\nReset of {reset_ctr} BN-layers successful. Recalibrating BN-statistics on {int(recalibration_fraction * 100)}% of the training data (unshuffled).\n"
        )
        n_batches = len(self.trainLoader)
        max_n_batches = int(recalibration_fraction * n_batches)
        if max_n_batches == 0:
            return
        it = 0
        with tqdm(self.trainLoader, leave=True) as pbar:
            for x_input, y_target, indices in pbar:
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)

                with autocast(enabled=(self.config.use_amp is True)):
                    self.model.train()(x_input)
                it += 1
                if it >= max_n_batches:
                    break

        torch.cuda.empty_cache()
        
        
        
    def init_quant_model(self,model, train_loader, device):
        model = model.to(device)
        for m in model.modules():
            if isinstance(m, QConv):
                m.init.data.fill_(1)

        iterloader = iter(train_loader)

        data = next(iterloader)
        if len(data) == 2:
            images, labels = data
        elif len(data) == 3:
            images, labels, indices = data  
        else:
            raise ValueError("Unexpected data format from train_loader.")
        

        images = images.to(device)

        model.train()
        model.forward(images)
        for m in model.modules():
            if isinstance(m, QConv):
                m.init.data.fill_(0)

    def update_grad_scales(self,model, train_loader, criterion, device, args):
        ## update scales
        if args.QActFlag:
            scaleA = []
        if args.QWeightFlag:
            scaleW = []
        for m in model.modules():
            if isinstance(m, QConv):
                m.hook_Qvalues = True
                if args.QActFlag:
                    scaleA.append(0)
                if args.QWeightFlag:
                    scaleW.append(0)

        model.train()
        with tqdm(total=3, file=sys.stdout) as pbar:
            for num_batches, (images, labels, indices) in enumerate(train_loader):
                if num_batches == 3: # estimate trace using 3 batches
                    break
                images = images.to(device)
                labels = labels.to(device)

                # forward with single batch
                model.zero_grad()
                pred = model(images)
                loss = criterion(pred, labels)
                loss.backward(create_graph=True)

                # store quantized values
                if args.QWeightFlag:
                    Qweight = []
                if args.QActFlag:
                    Qact = []
                for m in model.modules():
                    if isinstance(m, QConv):
                        if args.QWeightFlag:
                            Qweight.append(m.buff_weight)
                        if args.QActFlag:
                            Qact.append(m.buff_act)

                # update the scaling factor for activations
                if args.QActFlag:
                    params = []
                    grads = []
                    for i in range(len(Qact)): # store variable & gradients
                        params.append(Qact[i])
                        grads.append(Qact[i].grad)

                    for i in range(len(Qact)):
                        trace_hess_A = np.mean(self.trace(model, [params[i]], [grads[i]], self.device))
                        avg_trace_hess_A = trace_hess_A / params[i].view(-1).size()[0] # avg trace of hessian
                        scaleA[i] += (avg_trace_hess_A / (grads[i].std().cpu().item()*3.0))

                # update the scaling factor for weights
                if args.QWeightFlag:
                    params = []
                    grads = []
                    for i in range(len(Qweight)):
                        params.append(Qweight[i])
                        grads.append(Qweight[i].grad)

                    for i in range(len(Qweight)):
                        trace_hess_W = np.mean(self.trace(model, [params[i]], [grads[i]], self.device))
                        avg_trace_hess_W = trace_hess_W / params[i].view(-1).size()[0]
                        scaleW[i] += (avg_trace_hess_W / (grads[i].std().cpu().item()*3.0))
                pbar.update(1)


        if args.QActFlag:
            for i in range(len(scaleA)):
                scaleA[i] /= num_batches
                scaleA[i] = np.clip(scaleA[i], 0, np.inf)
            print("\n\nscaleA\n", scaleA)
        if args.QWeightFlag:
            for i in range(len(scaleW)):
                scaleW[i] /= num_batches
                scaleW[i] = np.clip(scaleW[i], 0, np.inf)
            print("scaleW\n", scaleW)       
        print("")

        i = 0
        for m in model.modules():
            if isinstance(m, QConv):
                if args.QWeightFlag:
                    m.bkwd_scaling_factorW.data.fill_(scaleW[i])
                if args.QActFlag:
                    m.bkwd_scaling_factorA.data.fill_(scaleA[i])
                m.hook_Qvalues = False
                i += 1
                
        torch.cuda.empty_cache()
                
                


    def group_product(self,xs, ys):
        """
        the inner product of two lists of variables xs,ys
        :param xs:
        :param ys:
        :return:
        """
        return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

    def hessian_vector_product(self,gradsH, params, v):
        """
        compute the hessian vector product of Hv, where
        gradsH is the gradient at the current point,
        params is the corresponding variables,
        v is the vector.
        """
        hv = torch.autograd.grad(gradsH,
                                 params,
                                 grad_outputs=v,
                                 only_inputs=True,
                                 retain_graph=True)
        return hv

    def trace(self,model, params, grads, device, maxIter=50, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1


            Hv = self.hessian_vector_product(grads, params, v)
            trace_vhv.append(self.group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def freeze_outside_boundary_weight_idx(self, boundaryRange=None , comment=""):
        if boundaryRange is None:
            boundaryRange = self.config.boundaryRange
        self.freeze_dict = {} 
        first_conv_found = False
        os.makedirs(run_plot_dir, exist_ok=True)
        for name, module in self.model.named_modules():

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue

            # Skip Quantization Params.
            if "quan" in name:
                continue

            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                continue

            if isinstance(module, nn.Linear):
                continue
            if ".downsample." in name:
                continue

            if hasattr(module, "weight") and module.weight is not None:

                w = module.weight.data   # float weight
                uW = module.uW.data      
                lW = module.lW.data      
                levels = module.weight_levels  
                w = (w - lW) / (uW - lW)
                w = w.clamp(min=0, max=1) # [0, 1]
                w = w *(self.config.act_levels -1) 

                w_numpy = w.detach().cpu().numpy().flatten()

                w_floor = torch.floor(w)
                w_ceil = torch.ceil(w)
                dist_floor = torch.abs(w - w_floor)
                dist_ceil = torch.abs(w - w_ceil)

                boundary_abs = boundaryRange 
                mask_floor = (dist_floor <= boundary_abs)
                mask_ceil = (dist_ceil <= boundary_abs)
                mask_near_int = mask_floor | mask_ceil
                freeze_mask = mask_near_int.float()
                self.freeze_dict[name] = freeze_mask
                
                num_frozen = freeze_mask.sum().item()
                total_elem = freeze_mask.numel()
                ratio = num_frozen / total_elem
                
                
                
    def print_freeze_ratios(self,freeze_dict):

        num = 0
        for name, freeze_mask in freeze_dict.items():

            num_total = freeze_mask.numel()
            num_frozen = freeze_mask.sum().item()
            freeze_ratio = num_frozen / num_total * 100

            print(f"[{name}] Freeze: {num_frozen}/{num_total} ({freeze_ratio:.2f}%)")
            num = num + 1
            if num == 3:
                break


    def random_unstructured_freeze_notzero_pruning(self):

        self.freeze_outside_boundary_weight_idx()
        first_conv_found = False
        freeze_dict = self.freeze_dict
        mask_dict      = {}   # {layer_name: Tensor}
        randval_dict   = {}   # {layer_name: Tensor}
        for name, module in self.model.named_modules():

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue

            # Skip Quantization Params.
            if "quan" in name:
                continue

            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                print("first_conv skip")
                continue

            if isinstance(module, nn.Linear):
                print("Linear Skip")
                continue
            if ".downsample." in name:
                print(f"Skipping downsample layer: {name}")
                continue


            elif hasattr(module, "weight") and module.weight is not None and name in freeze_dict:

                print("Pruning module name:", name)
                freeze_mask = freeze_dict[name]  # shape == module.weight.shape
                device = module.weight.device

                num_frozen = freeze_mask.sum().item()
                total_elem = freeze_mask.numel()
                ratio = num_frozen / total_elem
                print(f" -> Freeze ratio for layer={name} : {ratio*100:.2f}%")

                not_freeze_positions = (freeze_mask == 0).nonzero(as_tuple=False)  # shape [K, N]
                K = not_freeze_positions.size(0)
                if K == 0:
                    print("No pruneable param in", name)
                    continue
                k_prune = int(K * ratio)
                if k_prune <= 0:
                    continue

                selected_indices = torch.randperm(K, device=device)[:k_prune]
                to_prune_indices = not_freeze_positions[selected_indices]  # shape [k_prune, N]


                weight_shape = module.weight.shape
                random_mask = torch.ones(weight_shape, dtype=torch.float, device=device)

                # (4) set 0 at to_prune_indices
                random_mask[tuple(to_prune_indices.t())] = 0


                final_mask = torch.max(freeze_mask, random_mask)
                

                base_random_values = torch.rand_like(module.weight)

                if hasattr(module, "lW") and hasattr(module, "uW"):
                    lW_layer = module.lW.detach()
                    uW_layer = module.uW.detach()
                else:                              
                    lW_layer = module.weight.data.min()
                    uW_layer = module.weight.data.max()


                raw_random_values = torch.empty_like(module.weight).uniform_(lW_layer, uW_layer)
                mask_dict[name]    = final_mask.cpu()        
                randval_dict[name] = raw_random_values.cpu()

                with torch.no_grad():
                    new_weights = module.weight * final_mask + raw_random_values  * (1 - final_mask)
                    module.weight.copy_(new_weights)

                num_replaced = int((final_mask == 0).sum())

    def freeze_batchnorm(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    module.weight.requires_grad = False 
                if module.bias is not None:
                    module.bias.requires_grad = False 
    def unfreeze_batchnorm(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True   