
import json
import sys
import warnings
import time
from collections import OrderedDict
import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path 
from runners.baseRunner import baseRunner
from utilities.utilities import Utilities as Utils
from tqdm.auto import tqdm
from copy import deepcopy
import seaborn as sns
from torch.utils.data import Dataset
from torchvision import transforms
import random

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
from torchvision import datasets
class pretrainedRunner(baseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_run = None
        self.prune_dict_file = None

    def find_existing_model(self, filterDict):
        """
        Returns:
          checkpoint_file, optimizer_file, scheduler_file, seed, reference_run
        """
        api     = wandb.Api()
        entity  = wandb.run.entity
        if self.config.phase == 1:
            project = (
                f"test"
            )

            warmup_filter = deepcopy(filterDict)

            runs = api.runs(f"{entity}/{project}", filters=warmup_filter)
            print(runs)
            assert runs, "WarmupQAT run not found!"
            run = next(r for r in runs if r.state != "failed")
            ckpt_file = f"files/warmup_qat_model.pt"
            opt_file  = f"files/warmup_qat_optimizer.pt"
            sch_file  = f"files/warmup_qat_scheduler.pt"
            run.file(ckpt_file).download(root=self.tmp_dir)
            run.file(opt_file).download(root=self.tmp_dir)
            run.file(sch_file).download(root=self.tmp_dir)
            seed = run.config["seed"]
            return (
                os.path.join(self.tmp_dir, ckpt_file),
                os.path.join(self.tmp_dir, opt_file),
                os.path.join(self.tmp_dir, sch_file),
                seed,
                run,
            )




    def get_missing_config(self):
        missing_config_keys = [
            "momentum",
            "n_epochs_warmup",
            "n_epochs",
        ]  # Have to have n_epochs even though it might be specified, otherwise ALLR doesnt have this


        try:
            additional_dict = {
                "last_training_lr": self.reference_run.summary["final.learning_rate"],
                "final.test.accuracy": self.reference_run.summary["final.test"][
                    "accuracy"
                ],
                "final.train.accuracy": self.reference_run.summary["final.train"][
                    "accuracy"
                ],
                "final.train.loss": self.reference_run.summary["final.train"]["loss"],
            }
        except KeyError as e:
            print(f"WandB Summary Keys: {list(self.reference_run.summary.keys())}") 

            additional_dict = {
                "last_training_lr": self.reference_run.summary["final.learning_rate"],
                "final.test.accuracy": self.reference_run.summary[
                    "final.test.accuracy"
                ],
                "final.train.accuracy": self.reference_run.summary[
                    "final.train.accuracy"
                ],
                "final.train.loss": self.reference_run.summary["final.train.loss"],
            }
        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = self.reference_run.config.get(
                    key
                )  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict["final.test.accuracy"]
        self.trained_train_loss = additional_dict["final.train.loss"]
        self.trained_train_accuracy = additional_dict["final.train.accuracy"]


    def fill_strategy_information(self):

        f = self.reference_run.file("files/iteration-lr-dict.json").download(
            root=self.tmp_dir
        )
        with open(f.name) as json_file:
            loaded_dict = json.load(json_file)
            self.strategy.lr_dict = OrderedDict(loaded_dict)
        # Upload iteration-lr dict from self.strategy to be used during retraining
        Utils.dump_dict_to_json_wandb(
            dumpDict=self.strategy.lr_dict, name="iteration-lr-dict"
        )

        file_path = os.path.join(wandb.run.dir, "iteration-lr-dict.json")
        with open(file_path, "w") as f:
            json.dump(self.strategy.lr_dict, f)
        wandb.save(file_path)

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""

        filterDict = {

        }
        assert self.config.phase is not None
        assert self.config.split_val is not None, "split_val has to be specified."
        if self.config.ensemble_by not in [None, "None", "none"]:
            
            # We do not perform regular IMP
            assert self.config.ensemble_by in [
                "seed",
                "weight_decay",
                "retrain_length",
                "retrain_schedule",
            ] 

        if self.config.learning_rate is not None: 
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model."
            )

        (
        self.checkpoint_file,
        self.optimizer_file,
        self.scheduler_file,
        self.seed,
        self.reference_run,
        ) = self.find_existing_model(filterDict=filterDict) 
        
        wandb.config.update({"seed": self.seed})  # Push the seed to wandb
        seed = self.seed
        if self.config.ensemble_by == "seed":
            # We use a new seed for retraining depending on the true seed (self.seed) and the seed
            seed = (self.seed + self.config.split_val + int((os.getpid() + 1) * time.time())) % 2**32
            sys.stdout.write(f"Original seed {self.seed}, new seed {seed}.\n")
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True
        self.get_missing_config()  # Load keys that are missing in the config

        (
            self.trainLoader,
            self.valLoader,
            self.testLoader,
            self.trainLoader_unshuffled,
        ) = self.get_dataloaders()
        self.model = self.get_model( 
            reinit=True, temporary=True
        )  # Load the previous model
        self.test_model_on_cifar100_c(self.model, model_id=f"testtt")
        

        split = self.config.split_val
        lr_m = random.uniform(1e-2, 1e-3) #imagenet 
        lr_m = random.uniform(1e-4, 1e-3) #cifar_tiny

        lr_q = 1e-5
        wd_m = random.uniform(5e-5, 1e-4)


        print("lr : " , lr_m)
        print("wd_: " , wd_m)


        self.optimizer_m = torch.optim.Adam(self.model_params_m, lr=lr_m , weight_decay=wd_m)
        self.optimizer_q = torch.optim.Adam(self.model_params_q, lr=lr_q)
        self.scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_m, T_max=self.config.n_epochs_per_phase, eta_min=0.0)
        self.scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q, T_max=self.config.n_epochs_per_phase, eta_min=0.0) 

      

        self.strategy = self.define_strategy()
        self.strategy.set_to_finetuning_phase()
        self.strategy.after_initialization()  # To ensure that all parameters are properly set (Actually, Doing Nothing.)
        print(f"update_pruning :  == {self.config.update_pruning} ")
        if self.config.update_pruning =="True":
            if self.config.phase :
                self.freeze_outside_boundary_weight_idx(comment = "")

        self.fill_strategy_information()



        """
        Pruning Code Start
        """

        if self.config.update_pruning == "True":
            if self.config.phase :
                self.magnitude_unstructured_freeze_notzero_pruning()
                self.recalibrate_bn() 
        
        
        print(f"self.freeze_batchnorm_bool == {self.config.freeze_batchnorm_bool}")
        if self.config.freeze_batchnorm_bool == "True":
            self.freeze_batchnorm() 
        
        
        """
        Pruning Code End
        """

        self.print_current_sparsity()
        
        if self.config.update_pruning =="True":
            if self.config.phase :
                self.freeze_outside_boundary_weight_idx(comment ="")
        


        # Run the computations

        self.strategy.at_train_end() 

        self.strategy.final() 
        if self.config.update_pruning =="True":
            if self.config.phase :
                self.freeze_outside_boundary_weight_idx(comment = "")


        # Save model, optimizer, scheduler
        self.checkpoint_file = self.save_model(model_type="qat")
        wandb.summary["final_model_file"] = f"qat_model_{self.config.split_val}_{self.config.phase}.pt"
        wandb.save(os.path.join(wandb.run.dir, f"qat_model_{self.config.split_val}_{self.config.phase}.pt"))

        self.optimizer_file = self.save_optimizer(model_type="qat")
        wandb.summary["final_optimizer_file"] = f"qat_optimizer_{self.config.split_val}_{self.config.phase}.pt"
        wandb.save(os.path.join(wandb.run.dir, f"qat_optimizer_{self.config.split_val}_{self.config.phase}.pt"))

        self.scheduler_file = self.save_scheduler(model_type="qat")
        wandb.summary["final_scheduler_file"] = f"qat_scheduler_{self.config.split_val}_{self.config.phase}.pt"
        wandb.save(os.path.join(wandb.run.dir, f"qat_scheduler_{self.config.split_val}_{self.config.phase}.pt"))
        



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
        print(images_sev.dtype, images_sev.shape)
        print(np.min(images_sev), np.max(images_sev))
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

    def magnitude_unstructured_freeze_notzero_pruning(self):

        first_conv_found = False
        freeze_dict = self.freeze_dict
        mask_dict      = {}
        randval_dict   = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            if "quan" in name:
                continue
            if isinstance(module, nn.Conv2d) and not first_conv_found:
                first_conv_found = True
                continue
            if isinstance(module, nn.Linear):
                continue
            if ".downsample." in name:
                continue

            elif hasattr(module, "weight") and module.weight is not None and name in freeze_dict:

                print("Pruning module name:", name)
                freeze_mask = freeze_dict[name].to(module.weight.device)
                device = module.weight.device
                
                total_elem_layer = module.weight.numel()
                freeze_ratio = freeze_mask.sum().item() / total_elem_layer

                not_freeze_positions = (freeze_mask == 0).nonzero(as_tuple=False)
                K_actual = not_freeze_positions.size(0)

                if K_actual == 0:
                    continue

                k_prune_target = int(np.ceil(K_actual * self.config.pruning_ratio))
                k_prune = min(k_prune_target, K_actual)

                if k_prune <= 0:
                    continue
                prunable_weights_values = module.weight[freeze_mask == 0]

                indices_of_smallest = torch.argsort(torch.abs(prunable_weights_values))[:k_prune]

                to_prune_indices = not_freeze_positions[indices_of_smallest]

                selected_values = prunable_weights_values[(indices_of_smallest)]

                weight_shape = module.weight.shape
                pruning_mask = torch.ones(weight_shape, dtype=torch.float, device=device)


                pruning_mask[tuple(to_prune_indices.t())] = 0


                final_mask = torch.max(freeze_mask, pruning_mask)
                

                if hasattr(module, "lW") and hasattr(module, "uW"):
                    lW_layer, uW_layer = module.lW.detach(), module.uW.detach()
                else:
                    lW_layer, uW_layer = module.weight.data.min(), module.weight.data.max()

                raw_random_values = torch.empty_like(module.weight).uniform_(lW_layer, uW_layer)
                mask_dict[name]    = final_mask.cpu()
                randval_dict[name] = raw_random_values.cpu()


                with torch.no_grad():
                    new_weights = module.weight * final_mask + raw_random_values * (1 - final_mask)
                    module.weight.copy_(new_weights)

                num_replaced = int((final_mask == 0).sum())


    def _prune_fname(self):
        return f"split{1}_phase{self.config.phase}.pt"
    

    def save_prune_dict(self, mask_dict, randval_dict):

        fpath = self.prune_dir / self._prune_fname()
        torch.save({"mask_dict": mask_dict,
                    "randval_dict": randval_dict}, fpath)
        self.prune_dict_file = str(fpath)      
        return fpath

    def load_prund_dict_and_freeze_notzero_prunig(self):

        state = self.load_prune_dict()
        mask_dict, rand_dict = state["mask_dict"], state["randval_dict"]


        for name, module in self.model.named_modules():
            if name not in mask_dict or not hasattr(module, "weight"):
                continue
            m = mask_dict[name].to(module.weight.device)
            r = rand_dict[name].to(module.weight.device)
            assert m.shape == module.weight.shape, f"{name} shape mismatch"

            with torch.no_grad():
                module.weight.copy_(module.weight * m + r * (1 - m))
    
    def load_prune_dict(self):

        if self.prune_dict_file is None:
            self.prune_dict_file = str(self.prune_dir / self._prune_fname())
        print(f"Loading prune_dict from: {self.prune_dict_file}")
        return torch.load(self.prune_dict_file, map_location=self.device)

        