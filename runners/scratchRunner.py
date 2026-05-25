
import os
import sys
import time

import numpy as np
import torch
import wandb

from runners.baseRunner import baseRunner
from utilities.utilities import Utilities as Utils
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class scratchRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.artifact = None

        entity, project = wandb.run.entity, wandb.run.project
        self.initial_artifact_name = f""

    def find_existing_model(self):
        """Finds an existing wandb artifact and downloads the initial model file."""
        # Create a new artifact, this is idempotent, i.e. no artifact is created if this already exists
        try:
            self.artifact = wandb.run.use_artifact(
                f""
            )
            seed = self.artifact.metadata["seed"]
            self.artifact.download(root=self.tmp_dir)
            self.checkpoint_file = os.path.join(self.tmp_dir, "initial_model.pt")
            self.seed = seed

        except Exception as e:
            print(e)

        outputStr = (
            f"Found {self.initial_artifact_name} with seed {seed}"
            if self.artifact is not None
            else "Nothing found."
        )
        sys.stdout.write(
            f"Trying to find reference initial model in project: {outputStr}\n"
        )

    def run(self):
        """Function controlling the workflow of scratchRunner"""
        # If not existing, start a new model, otherwise use existing one with same run-id
        self.find_existing_model()

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


        # if Ident Quantizer

        (
            self.trainLoader,
            self.valLoader,
            self.testLoader,
            self.trainLoader_unshuffled,
        ) = self.get_dataloaders()
        self.model = self.get_model(reinit=True, temporary=True)
        # Save initial model before training

        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        initial_lr = 0.1 # From "ContinueCosine, 0.01"

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01, # This will be controlled by the scheduler
            momentum=0.9,
            weight_decay=0.0005
        )

        
        # Cosine Annealing Scheduler
        # It will anneal the learning rate from initial_lr down to eta_min (default 0)
        # over T_max epochs.
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_epochs, # Total number of epochs for one cycle
            eta_min=0.0 # Minimum learning rate (decays to 0.0)
        )

        # Use num_total_epochs for the loop range
        for epoch in range(self.config.total_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            correct_predictions_train = 0
            total_samples_train = 0

            # Training phase
            for batch_idx, (inputs, labels, *_) in enumerate(self.trainLoader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples_train += labels.size(0)
                correct_predictions_train += (predicted == labels).sum().item()

                # Use num_total_epochs in the print statement
                if (batch_idx + 1) % 50 == 0:  # Log every 100 batches
                    current_lr = self.optimizer.param_groups[0]['lr'] # Get current learning rate


            epoch_train_loss = running_loss / total_samples_train
            epoch_train_accuracy = 100.0 * correct_predictions_train / total_samples_train

            # Use num_total_epochs in the print statement
            print(f"Epoch [{epoch+1}/{self.config.total_epochs}] Training Completed: Avg Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_accuracy:.2f}%")

            # Validation phase
            if self.testLoader:
                self.model.eval()  # Set the model to evaluation mode
                val_loss = 0.0
                correct_predictions_val = 0
                total_samples_val = 0
                with torch.no_grad():  # No gradients needed for validation
                    for inputs, labels,_ in self.testLoader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total_samples_val += labels.size(0)
                        correct_predictions_val += (predicted == labels).sum().item()

                epoch_val_loss = val_loss / total_samples_val
                epoch_val_accuracy = 100.0 * correct_predictions_val / total_samples_val

                # Use num_total_epochs in the print statement
                print(f"Epoch [{epoch+1}/{self.config.total_epochs}] Validation: Avg Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.2f}%")


            self.scheduler.step()


        self.checkpoint_file = self.save_model(model_type="trained")
        
        wandb.summary["final_model_file"] = "trained_model.pt"

        
