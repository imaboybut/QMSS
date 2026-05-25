
import os
import sys
import time

import numpy as np
import torch
import wandb
import json

import importlib
from runners.baseRunner import baseRunner
from utilities.utilities import Utilities as Utils


class warmupQATRunner(baseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.artifact = None

        entity, project = wandb.run.entity, wandb.run.project
        self.initial_artifact_name = f""

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the pretrained full precision model file."""
        filterDict["$and"].append({"config.strategy": "Dense"}) 

        entity = wandb.run.entity
        import re

        arch_number = re.search(r'\d+', self.config.arch).group() 
        #project = ""
        project = ""
        
        print("Using Dense pretrained moodel")

        api = wandb.Api()
        # Some variables have to be extracted from the filterDict and checked manually, e.g. weight decay in scientific format
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

        checkpoint_file = None
        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        
        for run in runs:
            print(run.config)
            
            if run.state == "failed":
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


            checkpoint_file = "files/" + "trained_model.pt"
        


            try:

                if checkpoint_file is not None:
                    runsExist = True
                    run.file(checkpoint_file).download(root=self.tmp_dir)
                    seed = run.config["seed"]
                    reference_run = run
                    break
            except (
                Exception
            ) as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                checkpoint_file = None
        assert not (
            runsExist and checkpoint_file is None
        ), "Runs found, but none of them have a model available -> abort."
        outputStr = (
            f"Found {checkpoint_file} in run {run.name}"
            if checkpoint_file is not None
            else "Nothing found."
        )
        sys.stdout.write(
            f"Trying to find reference trained model in project: {outputStr}\n"
        )
        assert (
            checkpoint_file is not None
        ), "No reference trained model found, Aborting."
        return checkpoint_file, seed, reference_run

    def run(self):

        filterDict = {
            "$and": [
                {"config.run_id": self.config.run_id},
                # {"config.arch": self.config.arch},
                {"config.optimizer": self.config.optimizer},
            ]
        }
        self.checkpoint_file, self.seed, self.reference_run = self.find_existing_model( 
            filterDict=filterDict
        )


        if self.seed is None: 
            # Generate a random seed
            self.seed = int((os.getpid() + 1) * time.time()) % 2**32

        wandb.config.update({"seed": self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True


        assert self.config.w_bit is not None and self.config.a_bit is not None
        (
            self.trainLoader,
            self.valLoader,
            self.testLoader,
            self.trainLoader_unshuffled,
        ) = self.get_dataloaders()

        self.model = self.get_model(reinit=False, temporary=True)

        
        
        
        # Save initial model before training
        if self.artifact is None: 
            self.artifact = wandb.Artifact(
                self.initial_artifact_name, type="model", metadata={"seed": self.seed}
            )
            sys.stdout.write(f"Creating {self.initial_artifact_name}.\n")
            self.save_model(model_type="initial", temporary=True)
            self.artifact.add_file(f"{self.tmp_dir}/initial_model.pt")
            wandb.run.use_artifact(self.artifact)

        self.strategy = self.define_strategy() 

        self.strategy.after_initialization()

        self.strategy.at_train_begin()

        # Do proper training
        self.train()

        self.strategy.at_train_end()


        self.checkpoint_file = self.save_model(model_type="warmup_qat")
        wandb.summary["final_model_file"] = "warmup_qat_model.pt"
        wandb.save(os.path.join(wandb.run.dir, "warmup_qat_model.pt"))

        self.optimizer_file = self.save_optimizer(model_type="warmup_qat")
        wandb.summary["final_optimizer_file"] = "warmup_qat_optimizer.pt"
        wandb.save(os.path.join(wandb.run.dir, "warmup_qat_optimizer.pt"))

        self.scheduler_file = self.save_scheduler(model_type="warmup_qat")
        wandb.summary["final_scheduler_file"] = "warmup_qat_scheduler.pt"
        wandb.save(os.path.join(wandb.run.dir, "warmup_qat_scheduler.pt"))
        


        self.strategy.final()

        # Upload iteration-lr dict from self.strategy to be used during retraining
        Utils.dump_dict_to_json_wandb(
            dumpDict=self.strategy.lr_dict, name="iteration-lr-dict"
        )
        file_path = os.path.join(wandb.run.dir, "iteration-lr-dict.json")
        with open(file_path, "w") as f:
            json.dump(self.strategy.lr_dict, f)
        wandb.save(file_path) 
