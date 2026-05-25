
import itertools
import json
import math
import os
import time
import hashlib
import sys
import warnings
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from runners.baseRunner import baseRunner
from strategies import ensembleStrategies
from utilities.utilities import (
    Utilities as Utils,
    WorstClassAccuracy,
    CalibrationError,
    Candidate,
)
import seaborn as sns
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
from torchvision import datasets

class ensembleRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def find_multiple_existing_models(self, filterDict):
        """Finds existing wandb runs and downloads the model files."""
        current_phase = self.config.phase  # We are in the same phase
        filterDict["$and"].append({"config.phase": current_phase})

        
        if current_phase > 1: 
            # We need to specify the previous ensemble method as well
            filterDict["$and"].append(
                {"config.ensemble_method": self.config.ensemble_method}
            )

        filterDict["$and"].append({"config.ensemble_by": self.config.ensemble_by})
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        candidate_model_list = []

        manualVariables = ["weight_decay", "penalty", "group_penalty"]
        manVarDict = {}
        dropIndices = []
        for var in manualVariables:
            for i in range(len(filterDict["$and"])):
                entry = filterDict["$and"][i]
                s = f"config.{var}"
                if s in entry:
                    dropIndices.append(i)
                    manVarDict[var] = entry[s]
        for idx in reversed(sorted(dropIndices)):
            filterDict["$and"].pop(idx)


        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        print("filterDict : ", filterDict)
        print("runs : ", len(runs))
        flag = 0
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs: 
            if flag == self.config.n_splits_total:
                break
            if run.state != "finished":
                # Ignore this run
                continue
            # Check if run satisfies the manual variables
            conflict = False
            for var, val in manVarDict.items():
                if var in run.config and run.config[var] != val:
                    conflict = True
                    break
            if conflict:
                continue
            sys.stdout.write(f"Trying to access {run.name}.\n")

            for file in run.files():
                print(f"File name: {file.name}")
            
            checkpoint_file = "files/" + run.summary.get("final_model_file") 
            try:
                if checkpoint_file is not None:
                    runsExist = True
                    sys.stdout.write(
                        f"Downloading pruned model with split {run.config['ensemble_by']} value: {run.config['split_val']}.\n"
                    )
                    run.file(checkpoint_file).download(root=self.tmp_dir, replace=True)
                    self.seed = run.config["seed"]
                    candidate_id = run.config["split_val"]
                    candidate_model_list.append(
                        Candidate(
                            candidate_id=candidate_id,
                            candidate_file=os.path.join(self.tmp_dir, checkpoint_file),
                            candidate_run=run,
                        )
                    )
                    flag +=1
            except (
                Exception
            ) as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                checkpoint_file = None
                break

            optimizer_file = "files/" + run.summary.get("final_optimizer_file") 
            try:
                if optimizer_file is not None:
                    runsExist = True
                    sys.stdout.write(
                        f"Downloading optimizer file with split {run.config['ensemble_by']} value: {run.config['split_val']}.\n"
                    )
 
                    run.file(optimizer_file).download(root=self.tmp_dir, replace=True)

            except (
                Exception
            ) as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                optimizer_file = None
                break

            scheduler_file ="files/" +  run.summary.get("final_scheduler_file") 
            try:
                if scheduler_file is not None:
                    runsExist = True
                    sys.stdout.write(
                        f"Downloading scheduler file with split {run.config['ensemble_by']} value: {run.config['split_val']}.\n"
                    )
                    run.file(scheduler_file).download(root=self.tmp_dir, replace=True)

            except (
                Exception
            ) as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                scheduler_file = None
                break
                
                
        assert not (
            runsExist and checkpoint_file is None
        ), "Runs found, but one of them has no model available -> abort."
        outputStr = (
            f"Found {len(candidate_model_list)} pruned models with split vals {sorted([c.id for c in candidate_model_list])}"
            if checkpoint_file is not None
            else "Nothing found."
        )
        sys.stdout.write(
            f"Trying to find reference pruned models in project: {outputStr}\n"
        )
        assert (
            checkpoint_file is not None
        ), "One of the pruned models has no model file to download, Aborting."


        return candidate_model_list, optimizer_file, scheduler_file


    def transport_information(self, ref_run):
        missing_config_keys = [
            "momentum",
            "n_epochs_warmup",
            "n_epochs",
        ]  # Have to have n_epochs even though it might be specified, otherwise ALLR doesnt have this

        try: 
            additional_dict = {
                "last_training_lr": ref_run.summary["final.learning_rate"],
                "final.test.accuracy": ref_run.summary["final.test.accuracy"],
                "final.train.accuracy": ref_run.summary["final.train.accuracy"],
                "final.train.loss": ref_run.summary["final.train.loss"],
            }
        except:
            additional_dict = {
                "last_training_lr": ref_run.summary["final.learning_rate"],
                "final.test.accuracy": ref_run.summary["final.test"]["accuracy"],
                "final.train.accuracy": ref_run.summary["final.train"]["accuracy"],
                "final.train.loss": ref_run.summary["final.train"]["loss"],
            }

        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = ref_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict["final.test.accuracy"]
        self.trained_train_loss = additional_dict["final.train.loss"]
        self.trained_train_accuracy = additional_dict["final.train.accuracy"]

        # Get the wandb information about lr and fill the corresponding strategy dicts, which can then be used by rewinders

        f = ref_run.file("files/iteration-lr-dict.json").download(root=self.tmp_dir) 
        with open(f.name) as json_file:
            loaded_dict = json.load(json_file)
            lr_dict = OrderedDict(loaded_dict)
        # Upload iteration-lr dict from self.strategy to be used during retraining
        Utils.dump_dict_to_json_wandb(dumpDict=lr_dict, name="iteration-lr-dict")

    def load_ensemble_opt_sch(self, filterDict):
        api     = wandb.Api()
        entity  = wandb.run.entity
        default_project = wandb.run.project
        filterDict = filterDict
        filterDict["$and"].extend([
            {"config.phase":self.config.phase - 1},
            {"config.strategy":  "Ensemble"},
            {"config.split_val": self.config.split_val},
            {"config.run_id": self.config.run_id},
            {"config.n_epochs_per_phase": self.config.n_epochs_per_phase},
        ])
        runs = api.runs(f"{entity}/{default_project}", filters=filterDict)
        run  = next(r for r in runs if r.state != "failed")

        opt_file = f"files/{run.summary['final_optimizer_file']}"
        sch_file = f"files/{run.summary['final_scheduler_file']}"
        run.file(opt_file).download(root=self.tmp_dir)
        run.file(sch_file).download(root=self.tmp_dir)

        return (
            os.path.join(self.tmp_dir, opt_file),   # Phase1 QAT optimizer
            os.path.join(self.tmp_dir, sch_file),   # Phase1 QAT scheduler
        )


        
    def compute_cosine_similarity(self, tensor1, tensor2):
        flat1 = tensor1.view(-1).float()
        flat2 = tensor2.view(-1).float()
        return F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

    def compare_state_dicts(self, state_dict1, state_dict2):
        similarities = {}

        for key in state_dict1.keys():
            if key in state_dict2 and 'init' not in key:
                param1 = state_dict1[key]
                param2 = state_dict2[key]
                if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
                    similarities[key] = self.compute_cosine_similarity(param1, param2)

        return similarities
    def compute_l2_distance(self, tensor1, tensor2):
        flat1 = tensor1.view(-1).float()
        flat2 = tensor2.view(-1).float()
        return torch.norm(flat1 - flat2, p=2).item()

    def compare_state_dicts_l2(self, state_dict1, state_dict2):
        distances = {}

        for key in state_dict1.keys():
            if key in state_dict2 and 'init' not in key:
                param1 = state_dict1[key]
                param2 = state_dict2[key]
                if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
                    distances[key] = self.compute_l2_distance(param1, param2)

        return distances

    def compare_all_model_l2_distance(self, candidate_models):
        num_models = len(candidate_models)
        state_dicts = []


        for idx in range(num_models):
            model_checkpoint = candidate_models[idx].file  
            model = torch.load(model_checkpoint, map_location=torch.device("cpu"))


            state_dict = model["state_dict"] if "state_dict" in model else model
            state_dicts.append(state_dict)


        for i in range(num_models):
            for j in range(i + 1, num_models):
                distances = self.compare_state_dicts_l2(state_dicts[i], state_dicts[j])

                avg_l2 = sum(distances.values()) / len(distances)

    def print_quant_values(self, candidate_models, layer_key: str):

        base_key = layer_key[:-len(".weight")] if layer_key.endswith(".weight") else layer_key


        lw_key = f"{base_key}.lW"
        u_key  = f"{base_key}.uW"
        module_lw_key = f"module.{lw_key}"
        module_u_key = f"module.{u_key}"


        for idx, cand in enumerate(candidate_models, start=1):
            ckpt = torch.load(cand.file, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)


            lw = state_dict.get(lw_key) or state_dict.get(module_lw_key)
            uw = state_dict.get(u_key)  or state_dict.get(module_u_key)


            if lw is not None:
                lw_val = lw.detach().cpu()
                print(f"      lW – min: {lw_val.min():.4f}, max: {lw_val.max():.4f}, mean: {lw_val.mean():.4f}")
            else:
                print(f"      lW: NOT FOUND")

            if uw is not None:
                uw_val = uw.detach().cpu()
                print(f"      uW – min: {uw_val.min():.4f}, max: {uw_val.max():.4f}, mean: {uw_val.mean():.4f}")
            else:
                print(f"      uW: NOT FOUND")


    def print_batchnorm_stats(self, candidate_models, layer_key: str):
        bias_key = layer_key[:-len(".weight")] + ".bias" if layer_key.endswith(".weight") else layer_key + ".bias"
        module_weight_key = f"module.{layer_key}"
        module_bias_key   = f"module.{bias_key}"

        for idx, cand in enumerate(candidate_models, start=1):
            ckpt = torch.load(cand.file, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)

            w = state_dict.get(layer_key, None)
            if w is None:
                w = state_dict.get(module_weight_key, None)

            b = state_dict.get(bias_key, None)
            if b is None:
                b = state_dict.get(module_bias_key, None)

            print(f"  - Model {idx} (split={cand.id}):")
            if w is not None:
                w = w.detach().cpu()
                print(f"      γ (weight) – min: {w.min():.4f}, max: {w.max():.4f}, mean: {w.mean():.4f}")
            else:
                print("      γ (weight): NOT FOUND")

            if b is not None:
                b = b.detach().cpu()
                print(f"      β (bias)   – min: {b.min():.4f}, max: {b.max():.4f}, mean: {b.mean():.4f}")
            else:
                print("      β (bias): NOT FOUND")

    def compare_all_model_weights(self, candidate_models: List) -> None:

        num_models = len(candidate_models)
        if num_models < 2:
            return

        state_dicts = []


        for model_info in candidate_models:
            model_checkpoint = model_info.file 
            model = torch.load(model_checkpoint, map_location=torch.device("cpu"))
            state_dict = model.get("state_dict", model)
            state_dicts.append(state_dict)


        total_similarity_sum = 0.0
        successful_comparisons = 0

        for i in range(num_models):
            similarities_vs_others = []


            for j in range(num_models):
                if i == j:
                    continue

                similarities = self.compare_state_dicts(state_dicts[i], state_dicts[j])
                filtered = {
                    k: v for k, v in similarities.items()
                    if k.endswith("conv1.weight") or k.endswith("conv2.weight")
                }
                if filtered:
                    avg_sim_for_pair = sum(filtered.values()) / len(filtered)
                    similarities_vs_others.append(avg_sim_for_pair)


            num_other_models = len(similarities_vs_others)
            overall_avg_sim = sum(similarities_vs_others) / num_other_models

            total_similarity_sum += overall_avg_sim
            successful_comparisons += 1



        if successful_comparisons > 0:
            grand_total_average = total_similarity_sum / successful_comparisons


        

    def load_soup_model(self, ensemble_state_dict):

        fName = f"ensemble_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)
        torch.save(ensemble_state_dict, fPath)  # Save the state_dict

        self.checkpoint_file = fName

        # Actually load the model
        self.model = self.get_model(
            reinit=True, temporary=True
        )  # Load the ensembled model
    

    def evaluate_soup(self, data="val", ensemble_labels: torch.Tensor = None):

        AccuracyMeter = Accuracy(num_classes=self.n_classes).to(device=self.device)
        ECEMeter = CalibrationError(norm="l1").to(device=self.device)
        MCEMeter = CalibrationError(norm="max").to(device=self.device)
        WorstClassAccuracyMeter = WorstClassAccuracy(num_classes=self.n_classes).to(
            device=self.device
        )

        if data == "val":
            loader = self.valLoader
        elif data == "test":
            loader = self.testLoader
        elif data == "ood":
            loader = self.oodLoader
            if loader is None:
                sys.stdout.write(f"No OOD data found, skipping OOD evaluation.\n")
                return {}
        else:
            raise NotImplementedError
        

        ensemble_labels = None
        if ensemble_labels is not None:
            sys.stdout.write(
                f"Performing computation of prediction ensemble {data} accuracy.\n"
            )
        else:
            sys.stdout.write(f"Performing computation of soup {data} accuracy.\n")

        with tqdm(loader, leave=True) as pbar:
            for x_input, y_target, indices in pbar:
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)
                indices = indices.to(self.device, non_blocking=True)

                y_target = y_target.to(self.device, non_blocking=True)

                with autocast(enabled=(self.config.use_amp is True)):
                    output = self.model.train(mode=False)(x_input)
                    AccuracyMeter(output, y_target)
                    ECEMeter(output, y_target)
                    MCEMeter(output, y_target)
                    WorstClassAccuracyMeter(output, y_target)

        outputDict = {
            "accuracy": AccuracyMeter.compute().item(),
            "ece": ECEMeter.compute().item(),
            "mce": MCEMeter.compute().item(),
            "worst_class_accuracy": WorstClassAccuracyMeter.compute().item(),
        }
        return outputDict



    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        assert self.config.ensemble_by in [
            "seed",
            "weight_decay",
            "retrain_length",
            "retrain_schedule",
        ]
        assert self.config.n_splits_total is not None
        assert self.config.split_val is None

        # Find the reference run
        filterDict = {
            "$and": [
                {"config.run_id": self.config.run_id},
                {"config.arch": self.config.arch},
                {"config.optimizer": self.config.optimizer},
                {"config.n_epochs_per_phase": self.config.n_epochs_per_phase},
                {"config.n_phases": self.config.n_phases},
                {"config.retrain_schedule": self.config.retrain_schedule},
                {"config.strategy": "QAT"},
                #{"config.boundaryRange" : self.config.boundaryRange},
                
            ]
        }
        #filterDict["$and"].append({"config.split_val": {"$in": [1, 4]}})

        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model."
            )
            filterDict["$and"].append(
                {"config.learning_rate": self.config.learning_rate}
            )
        if self.config.n_epochs is not None:
            warnings.warn(
                "You specified n_epochs for retraining. Note that this only controls the selection of the pretrained model."
            )
            filterDict["$and"].append({"config.n_epochs": self.config.n_epochs})
        candidate_models, self.optimizer_file, self.scheduler_file = (
            self.find_multiple_existing_models(filterDict=filterDict)
        )

        
        for idx, model in enumerate(candidate_models):
            ckpt_path = model.file
            print(f"[{idx}] Model ID: {model.id}")
            print(f"    -> Checkpoint File: {ckpt_path}")

            if os.path.exists(ckpt_path):
                with open(ckpt_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                print(f"    -> MD5 Hash: {file_hash}")


        self.print_quant_values(candidate_models, layer_key="layer2.1.conv1.weight")
        self.print_batchnorm_stats(candidate_models,layer_key = "layer2.1.bn1.weight")
        self.compare_all_model_weights(candidate_models)
        self.compare_all_model_l2_distance(candidate_models)


        hamming_distance_matrix = self.count_pairwise_mapping_shifts(candidate_models)
        print(hamming_distance_matrix)


        self.seed = int((os.getpid() + 1) * time.time()) % 2**32
        wandb.config.update({"seed": self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True

        self.transport_information(ref_run=candidate_models[0].run)

        (
            self.trainLoader,
            self.valLoader,
            self.testLoader,
            self.trainLoader_unshuffled,
        ) = self.get_dataloaders()
        
        
        results = self.compute_connectivity_from_files(candidate_models, self.testLoader, self.loss_criterion, self.device)
        print(results)
        self.save_connectivity_curves(results)
        


        if self.config.n_splits_total == 20:
            print(f"\nFound {len(candidate_models)} candidate models. Starting individual evaluation on CIFAR-100-C.")
            for idx, model_info in enumerate(candidate_models):
                print(f"\n--- [{idx+1}/{len(candidate_models)}] Evaluating Model ID: {model_info.id} ---")
                print(f"  -> Checkpoint: {model_info.file}")


                model_instance = self.get_model(reinit=True, temporary=True)
                state_dict = torch.load(model_info.file, map_location=self.device)
                model_instance.load_state_dict(state_dict)
                model_instance.to(self.device)
                model_instance.eval()

                #self.recalibrate_bn_on_model(model_instance)

                
                #self.test_model_on_tiny_imagenet_c(model_instance, model_id=f"model_{idx}_{model_info.id}")
                self.test_model_on_cifar100_c(model_instance, model_id=f"model_{idx}_{model_info.id}")

            print("\n--- Individual model evaluation on CIFAR-100-C finished. ---")

        print("\n[FQ Metrics] Starting FQ diversity calculation for candidate models...")
        self.compute_and_log_fq_diversity(candidate_models, self.testLoader, self.device)
        print("[FQ Metrics] FQ diversity calculation finished.")
        

        callbackDict = {
            "final_log_callback": self.final_log,
            "soup_evaluation_callback": self.evaluate_soup,
            "load_soup_callback": self.load_soup_model,
            "recalibrate_bn_callback": self.recalibrate_bn,
        }
        self.ensemble_strategy = getattr(
            ensembleStrategies, self.config.ensemble_method 
        )(
            model=None,
            n_classes=self.n_classes,
            config=self.config,
            candidate_models=candidate_models,
            runner=self,
            callbacks=callbackDict,
        )

        #self.ensemble_strategy.collect_candidate_information()

        # Create ensemble
        ensemble_state_dict = self.ensemble_strategy.create_ensemble() 

        # Save the ensemble state dict
        fName = f"ensemble_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)
        torch.save(ensemble_state_dict, fPath)  # Save the state_dict
        self.checkpoint_file = fName

        
        if self.config.phase > 1 and self.config.n_epoch_soup_after_finetune > 0:
            filterDict = {
                "$and": []
            }
            (self.optimizer_file, self.scheduler_file) =self.load_ensemble_opt_sch(filterDict=filterDict)
        # Actually load the modeln_epochs_finetu
        self.model = self.get_model(
            reinit=True, temporary=True
        )  # Load the ensembled model

        

        
        self.strategy = self.define_strategy(use_dense_base=True)
        self.strategy.after_initialization()
        
        self.count_quantization_mapping_shifts(candidate_models, self.model)

        self.ensemble_strategy.final()

        self.test_model_on_cifar100_c(self.model, model_id=f"soup model{self.config.n_splits_total}")


        self.checkpoint_file = self.save_model(model_type="ensemble")
        wandb.summary["final_model_file"] = (f"ensemble_model_{self.config.ensemble_method}_{self.config.phase}.pt")
        wandb.save(os.path.join(wandb.run.dir, f"ensemble_model_{self.config.ensemble_method}_{self.config.phase}.pt"))

        self.optimizer_file = self.save_optimizer(model_type="ensemble")
        wandb.summary["final_optimizer_file"] = (f"ensemble_optimizer_{self.config.ensemble_method}_{self.config.phase}.pt")
        wandb.save(os.path.join(wandb.run.dir, f"ensemble_optimizer_{self.config.ensemble_method}_{self.config.phase}.pt"))

        self.scheduler_file = self.save_scheduler(model_type="ensemble")
        wandb.summary["final_scheduler_file"] = (f"ensemble_scheduler_{self.config.ensemble_method}_{self.config.phase}.pt")
        wandb.save(os.path.join(wandb.run.dir, f"ensemble_scheduler_{self.config.ensemble_method}_{self.config.phase}.pt"))
        
        
    def test_model_on_tiny_imagenet_c(self, model, model_id):
        corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        severities = [1, 2, 3, 4, 5]
        all_accuracies = []

        for corruption in tqdm(corruption_types, desc=f"Testing Corruptions for {model_id}"):
            for severity in severities:
                try:
                    c_loader = self.get_tiny_imagenet_c_loader(corruption, severity)
                    accuracy = self._evaluate_accuracy(model, c_loader, self.device)

                    
                    all_accuracies.append(accuracy)
                except FileNotFoundError:
                    print(f"Warning: Data for {corruption} severity {severity} not found. Skipping.")
                    continue

        if all_accuracies:
            mean_corruption_accuracy = np.mean(all_accuracies)
            print(f"[{model_id}] Mean Corruption Accuracy (MCA): {mean_corruption_accuracy:.4f}%")


            wandb.summary[f"MCA_{model_id}"] = mean_corruption_accuracy

    def get_tiny_imagenet_c_loader(self, corruption_type, severity):

        data_dir = "/data/Tiny-ImageNet-C"


        corruption_dir = os.path.join(data_dir, corruption_type, str(severity))

        if not os.path.isdir(corruption_dir):
            raise FileNotFoundError(f"Directory for corruption '{corruption_type}' severity {severity} not found in {data_dir}")
        
        transf = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ])
            

        dataset = datasets.ImageFolder(root=corruption_dir, transform=transf)
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        return loader

    def _evaluate_accuracy(self, model, loader, device):

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    def plot_heatmap(self,distance_matrix, title="Pairwise Hamming Distance", save_path=None,
                     show_lower_triangle=True):
        """
        Visualize a pairwise distance matrix as a heatmap, optionally masking one triangle.

        Args:
            title (str): Plot title.
            save_path (str | None): If provided, path to save the figure.
            show_lower_triangle (bool): If True, keep lower-triangle + diagonal and mask upper;
                                        if False, keep upper-triangle + diagonal.
        """

        distance_matrix = np.asarray(distance_matrix, dtype=float)

        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix must be a square (n×n) array.")

        n = distance_matrix.shape[0]              

        if show_lower_triangle:                    
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        else:                                     
            mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

        plt.figure(figsize=(8, 7))
        sns.heatmap(
            distance_matrix,
            mask=mask,
            cmap="viridis",
            linewidths=0.5,
            vmin=0.0,
            vmax=0.3,
            square=True,
            cbar_kws={"label": "Hamming distance"}
        )

        plt.title(title, fontsize=10)
        plt.xlabel("Candidate Model Index", fontsize=9)
        plt.ylabel("Candidate Model Index", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    def recalibrate_bn_on_model(self, model_to_recalibrate):

        model_to_recalibrate.train()

        recalibration_fraction = self.config.bn_recalibration_frac
        if recalibration_fraction is None or not (0 <= recalibration_fraction <= 1):
            recalibration_fraction = 1.0

        reset_ctr = 0
        for m in model_to_recalibrate.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.reset_running_stats()
                reset_ctr += 1
        if reset_ctr == 0:
            model_to_recalibrate.eval()
            return

        n_batches = len(self.trainLoader_unshuffled)
        max_n_batches = int(recalibration_fraction * n_batches)
        if max_n_batches == 0:
            model_to_recalibrate.eval()
            return

        with torch.no_grad():
            with tqdm(self.trainLoader_unshuffled, leave=False, desc="Re-BN ") as pbar:
                it = 0
                for x_input, _, _ in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)

                    with autocast(enabled=(self.config.use_amp is True)):
                        _ = model_to_recalibrate(x_input)

                    it += 1
                    if it >= max_n_batches:
                        break


        model_to_recalibrate.eval()
        torch.cuda.empty_cache()
    
    def evaluate_model(self,model, dataloader, criterion, device):
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, target, _  in dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                _, pred = outputs.max(1)
                total_correct += pred.eq(target).sum().item()
                total_samples += data.size(0)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def compute_connectivity_from_files(self,candidate_models, val_loader, criterion, device, alphas=np.linspace(0, 1, 11)):

        connectivity_results = {}
        candidate_state_dicts = []
        candidate_performances = []
        

        for idx, candidate in enumerate(candidate_models):
            ckpt_path = candidate.file 
            print(f"Loading checkpoint for candidate {candidate.id} from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            candidate_state_dicts.append(state_dict)


            model_instance = self.get_model(reinit=True, temporary=True)
            model_instance.load_state_dict(state_dict)
            loss, acc = self.evaluate_model(model_instance, val_loader, criterion, device)
            candidate_performances.append((loss, acc))
            print(f"Candidate {candidate.id}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")


        for i in range(len(candidate_models)):
            for j in range(i + 1, len(candidate_models)):
                sd1 = candidate_state_dicts[i]
                sd2 = candidate_state_dicts[j]
                loss1, acc1 = candidate_performances[i]
                loss2, acc2 = candidate_performances[j]

                interp_loss_list = []
                interp_acc_list = []
                weighted_loss_list = []
                weighted_acc_list = []


                for alpha in alphas:

                    interp_sd = {key: alpha * sd1[key] + (1 - alpha) * sd2[key] for key in sd1.keys()}


                    model_interp = self.get_model(reinit=True, temporary=True)
                    model_interp.load_state_dict(interp_sd)
                    loss_interp, acc_interp = self.evaluate_model(model_interp, val_loader, criterion, device)

                    interp_loss_list.append(loss_interp)
                    interp_acc_list.append(acc_interp)
                    weighted_loss_list.append(alpha * loss1 + (1 - alpha) * loss2)
                    weighted_acc_list.append(alpha * acc1 + (1 - alpha) * acc2)


                loss_barrier = max([interp_loss_list[k] - weighted_loss_list[k] for k in range(len(alphas))])

                acc_barrier = max([1 - (interp_acc_list[k] / weighted_acc_list[k]) if weighted_acc_list[k] > 0 else 0 for k in range(len(alphas))])


                pair_key = (candidate_models[i].id, candidate_models[j].id)
                connectivity_results[pair_key] = {
                    "loss_barrier": loss_barrier,
                    "alphas": alphas, 
                    "interp_losses": interp_loss_list, 
                    "interp_accuracies": interp_acc_list 
                }
                # ===============================
                print(f"Connectivity between {candidate_models[i].id} and {candidate_models[j].id}: "
                      f"Loss Barrier = {loss_barrier:.4f}, Accuracy Barrier = {acc_barrier:.4f}")
                

        return connectivity_results

    def save_connectivity_curves(self, results, folder_name="connectivity"):

        os.makedirs(folder_name, exist_ok=True)

        plt.figure(figsize=(4, 3))  

        for pair_key, data in results.items():
            model_id_1, model_id_2 = pair_key
            alphas = data['alphas']
            accs = data['interp_accuracies']
            plt.plot(alphas, accs, marker='o', linewidth=1.5, label=f'{model_id_1}-{model_id_2}')

        plt.xlabel(r'Interpolation Coefficient $\alpha$', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=8, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

        save_path = os.path.join(folder_name, f"connectivity_curve_{self.config.pruning_ratio}.pdf")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Connectivity curve saved to: {save_path}")
    def random_unstructured_pruning(self):

        first_conv_found = False
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            if "quan" in name:
                continue

            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                print("first_conv skip")
                continue

            if isinstance(module, nn.Linear):
                print("Linear Skip")
                continue

            if hasattr(module, "weight") and module.weight is not None:
                print("Pruning module name : {}".format(name))
                prune.random_unstructured(
                    module, name="weight", amount=self.config.pruning_ratio
                )

    def magnitude_unstructured_pruning(self):
        first_conv_found = False
        for name, module in self.model.named_modules():

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue

            if "quan" in name:
                continue

            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                print("first_conv skip")
                continue

            if isinstance(module, nn.Linear):
                print("Linear Skip")
                continue

            if hasattr(module, "weight") and module.weight is not None:
                print("Pruning module name : {}".format(name))
                prune.l1_unstructured(
                    module, name="weight", amount=self.config.pruning_ratio
                )

    def reinitialize_pruned_weights(model):

        first_conv_found = False
        for name, module in model.named_modules():
            if (
                isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                or "quan" in name
                or (isinstance(module, nn.Conv2d) and not first_conv_found)
                or isinstance(module, nn.Linear)
            ):

                if isinstance(module, nn.Conv2d) and not first_conv_found:
                    first_conv_found = True
                print(f"Skipping module: {name}")
                continue

            if hasattr(module, "weight") and module.weight is not None:
                if hasattr(module, "weight_mask"):
                    print(f"Reinitializing pruned weights for module: {name}")
                    mask = module.weight_mask

                    with torch.no_grad():
                        try:
                            new_weights = torch.empty_like(module.weight)
                            nn.init.kaiming_uniform_(new_weights)
                            module.weight.data[mask == 0] = new_weights[mask == 0]
                            prune.remove(module, "weight")
                        except Exception as e:
                            print(f"Error reinitializing module {name}: {str(e)}")

        print("Reinitialization complete.")

    def make_pruning_permanent(self):
        first_conv_found = False
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                print("Skip bn module name : {}".format(name))
                continue

            # Skip Quantization Params.
            if "quan" in name:
                print("Skip Quant module name : {}".format(name))
                continue

            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                print("first_conv skip")
                continue

            if isinstance(module, nn.Linear):
                print("Linear Skip")
                continue

            if hasattr(module, "weight") and module.weight is not None:
                print("Pruning permanent module name : {}".format(name))
                prune.remove(module, "weight")

    def print_current_sparsity(self):
        total_params = 0
        zero_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue

            # Skip Quantization Params.
            if "quan" in name:
                continue

            if hasattr(module, "weight") and module.weight is not None:
                total_params += module.weight.nelement()
                zero_params += torch.sum(module.weight == 0).item()

        sp = zero_params / total_params if total_params > 0 else 0
        print("Current model's sparsity : {}".format(sp))

        
    def _get_integer_mapping_from_qconv(self, weights, lW, uW, num_levels):
        normalized_weights = (weights - lW) / (uW - lW)
        normalized_weights = torch.clamp(normalized_weights, min=0, max=1)
        
        integer_map = torch.round(normalized_weights * (num_levels - 1))
        
        return integer_map.to(torch.int8) 
    def count_pairwise_mapping_shifts(self, candidate_models):


        num_models = len(candidate_models)

        distance_matrix = np.zeros((num_models, num_models), dtype=np.float32)

        for model_a_idx, model_a_info in enumerate(candidate_models):
            for model_b_idx, model_b_info in enumerate(candidate_models[model_a_idx + 1:], start=model_a_idx + 1):



                total_shifted_count = 0
                total_weight_count = 0

                try:
                    model_a = self.get_model(reinit=True, temporary=True)
                    ckpt_a = torch.load(model_a_info.file, map_location="cpu")
                    model_a.load_state_dict(ckpt_a.get('state_dict', ckpt_a), strict=False)
                    model_a.eval()
                    modules_a = dict(model_a.named_modules())

                    model_b = self.get_model(reinit=True, temporary=True)
                    ckpt_b = torch.load(model_b_info.file, map_location="cpu")
                    model_b.load_state_dict(ckpt_b.get('state_dict', ckpt_b), strict=False)
                    model_b.eval()
                    modules_b = dict(model_b.named_modules())


                first_conv_found = False
                for layer_name, module_b in modules_b.items():

                    if isinstance(module_b, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        continue
                    if "quan" in layer_name:
                        continue
                    if isinstance(module_b, torch.nn.Conv2d) and not first_conv_found:
                        first_conv_found = True

                        continue
                    if isinstance(module_b, torch.nn.Linear):
                        continue
                    if ".downsample." in layer_name:
                        continue


                    if hasattr(module_b, "weight") and module_b.weight is not None:
                        try:
                            if layer_name not in modules_a:
                                continue
                            module_a = modules_a[layer_name]

                            W_b = module_b.weight.data.float().cpu()
                            lW_b, uW_b = module_b.lW.data.cpu(), module_b.uW.data.cpu()
                            levels_b = module_b.weight_levels

                            W_a = module_a.weight.data.float().cpu()
                            lW_a, uW_a = module_a.lW.data.cpu(), module_a.uW.data.cpu()
                            levels_a = module_a.weight_levels

                            Q_a = self._get_integer_mapping_from_qconv(W_a, lW_a, uW_a, levels_a)
                            Q_b = self._get_integer_mapping_from_qconv(W_b, lW_b, uW_b, levels_b)

                            num_changed = torch.sum(Q_a != Q_b).item()
                            total_in_layer = W_b.numel()

                            total_shifted_count += num_changed
                            total_weight_count += total_in_layer
                        except AttributeError:
                            pass

                print("-" * 50)
                if total_weight_count > 0:

                    relative_distance = total_shifted_count / total_weight_count
                    distance_matrix[model_a_idx, model_b_idx] = relative_distance
                    distance_matrix[model_b_idx, model_a_idx] = relative_distance 

                    percentage = relative_distance * 100


                print("-" * 50)


        return distance_matrix
        
        
        
    def calculate_pairwise_hamming_distance(self, candidate_models):

        num_models = len(candidate_models)
        if num_models < 2:
            print("  -> Need at least 2 models to compare. Skipping.")
            return None


        distance_matrix = np.zeros((num_models, num_models), dtype=int)
        device = "cuda" if torch.cuda.is_available() else "cpu" 


        for i in range(num_models):
            for j in range(i + 1, num_models): 
                model_info_i = candidate_models[i]
                model_info_j = candidate_models[j]



                try:
                    model_i = self.get_model(reinit=True, temporary=True)
                    state_dict_i = torch.load(model_info_i.file, map_location=device).get("state_dict")
                    model_i.load_state_dict(state_dict_i, strict=False)
                    model_i.eval()
                    modules_i = dict(model_i.named_modules())

                    model_j = self.get_model(reinit=True, temporary=True)
                    state_dict_j = torch.load(model_info_j.file, map_location=device).get("state_dict")
                    model_j.load_state_dict(state_dict_j, strict=False)
                    model_j.eval()
                    modules_j = dict(model_j.named_modules())

                except Exception as e:
                    print(f"  -> Error loading models for pair ({i}, {j}): {e}. Skipping.")
                    continue

                total_shifted_count = 0
                total_weight_count = 0

                for layer_name, module_i in modules_i.items():

                    if isinstance(module_i, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        continue
                    if "quan" in layer_name:
                        continue
                    if isinstance(module_i, torch.nn.Conv2d) and not first_conv_found:
                        first_conv_found = True
                        continue
                    if isinstance(module_i, torch.nn.Linear):
                        continue
                    if ".downsample." in layer_name:
                        continue


                    
                    if hasattr(module_i, "weight") and module_i.weight is not None:
                        if layer_name in modules_j and hasattr(modules_j[layer_name], "weight"):
                            module_j = modules_j[layer_name]

                            Q_i = self._get_integer_mapping_from_qconv(module_i.weight.data, module_i.lW, module_i.uW, module_i.weight_levels)
                            Q_j = self._get_integer_mapping_from_qconv(module_j.weight.data, module_j.lW, module_j.uW, module_j.weight_levels)

                            num_changed = torch.sum(Q_i != Q_j).item()
                            total_shifted_count += num_changed
                            total_weight_count += Q_i.numel()


                distance_matrix[i, j] = total_shifted_count
                distance_matrix[j, i] = total_shifted_count 

                if total_weight_count > 0:
                    percentage = (total_shifted_count / total_weight_count) * 100



        return distance_matrix

    def test_model_on_cifar100_c(self, model, model_id):

        corruption_types = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 
            'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 
            'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
        ]
        severities = [1, 2, 3, 4, 5]
        all_accuracies = []

        for corruption in tqdm(corruption_types, desc=f"Testing Corruptions for {model_id}"):
            for severity in severities:
                try:
                    c_loader = self.get_cifar100_c_loader(corruption, severity)
                    accuracy = self._evaluate_accuracy(model, c_loader, self.device)
                    all_accuracies.append(accuracy) 
                except FileNotFoundError:
                    print(f"Warning: Data for {corruption} severity {severity} not found. Skipping.")
                    continue

        if all_accuracies:
            mean_corruption_accuracy = np.mean(all_accuracies)
            print(f"[{model_id}] Mean Corruption Accuracy (MCA): {mean_corruption_accuracy:.4f}%")


            wandb.summary[f"MCA_{model_id}"] = mean_corruption_accuracy


    def get_cifar100_c_loader(self, corruption_type, severity):

        data_dir = "/data/CIFAR-100-C"


        images_path = os.path.join(data_dir, corruption_type + '.npy')
        labels_path = os.path.join(data_dir, 'labels.npy')

        if not (os.path.exists(images_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(f"Data files for corruption '{corruption_type}' not found in {data_dir}")

        images = np.load(images_path)
        labels = np.load(labels_path)


        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        images_sev = images[start_idx:end_idx]
        labels_sev = labels[start_idx:end_idx]


        from torchvision.transforms import ToTensor, Normalize
        from PIL import Image


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


        tensor_images = torch.stack([transform(Image.fromarray(img)) for img in images_sev])

        tensor_labels = torch.from_numpy(labels_sev).long()

        dataset = TensorDataset(tensor_images, tensor_labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        return loader


    def _evaluate_accuracy(self, model, loader, device):

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total



    def compute_and_log_fq_diversity(self, candidate_models, loader, device):

        all_model_predictions, y_true = self._get_all_predictions(candidate_models, loader, device)

        fq_bd = self._calculate_fq_bd(all_model_predictions, y_true)
        fq_kw = self._calculate_fq_kw(all_model_predictions, y_true)
        fq_gd = self._calculate_fq_gd(all_model_predictions, y_true)

        print(f"FQ-BD: {fq_bd:.4f}, FQ-KW: {fq_kw:.4f}, FQ-GD: {fq_gd:.4f}")

        wandb.summary["FQ_Diversity/BD"] = fq_bd
        wandb.summary["FQ_Diversity/KW"] = fq_kw
        wandb.summary["FQ_Diversity/GD"] = fq_gd
        
        return {"FQ-BD": fq_bd, "FQ-KW": fq_kw, "FQ-GD": fq_gd}

    def _get_all_predictions(self, candidate_models, loader, device):

        all_model_predictions_list = []
        y_true = None

        for candidate in tqdm(candidate_models, desc="[FQ Metrics] Gathering Predictions"):
            ckpt_path = candidate.file
            state_dict = torch.load(ckpt_path, map_location=device)
            
            model_instance = self.get_model(reinit=True, temporary=True)
            model_instance.load_state_dict(state_dict)

            preds, labels = self._get_predictions_from_single_model(model_instance, loader, device)
            all_model_predictions_list.append(preds)
            
            if y_true is None:
                y_true = labels
        
        return np.array(all_model_predictions_list), y_true

    def _get_predictions_from_single_model(self, model, loader, device):

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_labels)

    @staticmethod
    def _calculate_fq_metric_base(all_predictions, y_true, base_metric_func):

        num_models = all_predictions.shape[0]
        focal_scores = []

        for i in range(num_models):
            focal_model_predictions = all_predictions[i, :]
            error_indices = np.where(focal_model_predictions != y_true)[0]

            if len(error_indices) == 0:
                continue

            M_focal = all_predictions[:, error_indices]
            y_true_focal = y_true[error_indices]

            score = base_metric_func(M_focal, y_true_focal)
            focal_scores.append(score)

        return np.mean(focal_scores) if focal_scores else 0.0

    def _calculate_fq_bd(self, all_predictions, y_true):

        return 1.0 - self._calculate_fq_metric_base(all_predictions, y_true, self._binary_disagreement)

    def _calculate_fq_kw(self, all_predictions, y_true):

        return 1.0 - self._calculate_fq_metric_base(all_predictions, y_true, self._kohavi_wolpert_variance)

    def _calculate_fq_gd(self, all_predictions, y_true):

        return 1.0 - self._calculate_fq_metric_base(all_predictions, y_true, self._generalized_diversity)

    @staticmethod
    def _binary_disagreement(M, y_true):
        Qs = []
        num_models, num_samples = M.shape
        if num_samples == 0: return 0.0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                N_0_1 = np.sum(np.logical_and(M[i, :] != y_true, M[j, :] == y_true))
                N_1_0 = np.sum(np.logical_and(M[i, :] == y_true, M[j, :] != y_true))
                Qs.append((N_0_1 + N_1_0) / num_samples)
        return np.mean(Qs) if Qs else 0.0

    @staticmethod
    def _kohavi_wolpert_variance(M, y_true):
        N, L = M.shape[1], M.shape[0]
        if N == 0 or L == 0: return 0.0
        kw = 0.
        for j in range(N):
            l_zj = np.sum(M[:, j] == y_true[j])
            kw += l_zj * (L - l_zj)
        return kw / (N * L * L)

    @staticmethod
    def _generalized_diversity(M, y_true):
        N, L = M.shape[1], M.shape[0]
        if N == 0 or L <= 1: return 0.0
        
        pi = np.zeros(L + 1)
        for i in range(N):
            pIdx = np.sum(M[:, i] != y_true[i])
            pi[pIdx] += 1
        
        pi /= N
        
        p1_denom = L
        p2_denom = L * (L - 1)
        if p1_denom == 0 or p2_denom == 0: return 0.0

        p1 = np.sum(np.arange(L + 1) * pi) / p1_denom
        p2 = np.sum(np.arange(L + 1) * (np.arange(L + 1) - 1) * pi) / p2_denom
        
        return (1.0 - p2 / p1) if p1 > 1e-8 else 0.0
    
    
    
    def count_quantization_mapping_shifts(self, candidate_models, ensemble_model):


        if not candidate_models:
            print("  -> No candidate models provided. Skipping.")
            return

        device = next(ensemble_model.parameters()).device
        ensemble_model.eval()
        ensemble_modules = dict(ensemble_model.named_modules())

        overall_total_shifted = 0
        overall_total_weights = 0

        for ref_idx, ref_model_info in enumerate(candidate_models):
            
            print(f"\n--- Comparing Ensemble vs Candidate {ref_idx} (ID: {ref_model_info.id}) ---")

            total_shifted_count = 0
            total_weight_count = 0

            try:
                ref_model = self.get_model(reinit=True, temporary=True) 
                ref_ckpt = torch.load(ref_model_info.file, map_location=device)
                state_dict = ref_ckpt.get("state_dict", ref_ckpt)
                ref_model.load_state_dict(state_dict, strict=False)
                ref_model.eval()
            except Exception as e:
                print(f"  -> Error loading reference model {ref_idx}: {e}. Skipping this comparison.")
                continue

            ref_modules = dict(ref_model.named_modules())
            first_conv_found = False 


            for layer_name, module_soup in ensemble_modules.items():
                

                if isinstance(module_soup, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    continue
                if "quan" in layer_name:
                    continue
                if isinstance(module_soup, torch.nn.Conv2d) and not first_conv_found:
                    first_conv_found = True

                    continue
                if isinstance(module_soup, torch.nn.Linear):
                    continue
                if ".downsample." in layer_name:
                    continue


                if hasattr(module_soup, "weight") and module_soup.weight is not None:
                    try:
                        if layer_name not in ref_modules:
                            continue
                        module_ref = ref_modules[layer_name]


                        W_soup = module_soup.weight.data.float().cpu()
                        lW_soup, uW_soup = module_soup.lW.data.cpu(), module_soup.uW.data.cpu()
                        levels_soup = module_soup.weight_levels

                        W_A = module_ref.weight.data.float().cpu() 
                        lW_A, uW_A = module_ref.lW.data.cpu(), module_ref.uW.data.cpu()
                        levels_A = module_ref.weight_levels


                        Q_A = self._get_integer_mapping_from_qconv(W_A, lW_A, uW_A, levels_A)
                        Q_soup = self._get_integer_mapping_from_qconv(W_soup, lW_soup, uW_soup, levels_soup)


                        num_changed = torch.sum(Q_A != Q_soup).item() 
                        total_in_layer = W_soup.numel() 



                        total_shifted_count += num_changed 
                        total_weight_count += total_in_layer

                    except AttributeError:
                        pass 
                    except Exception as e:
                        print(f"  -> Error processing layer [{layer_name}] for ref {ref_idx}: {e}. Skipping.")


            print("-" * 50)
            if total_weight_count > 0:
                percentage = (total_shifted_count / total_weight_count) * 100
            
                overall_total_shifted += total_shifted_count
                overall_total_weights += total_weight_count


        print("\n" + "=" * 60)
        if overall_total_weights > 0:
            overall_percentage = (overall_total_shifted / overall_total_weights) * 100

