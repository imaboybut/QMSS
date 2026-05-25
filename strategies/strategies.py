
import sys
from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune


#### Dense Base Class
class Dense:
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits."""

    required_params = []

    def __init__(self, **kwargs):
        self.masks = dict()
        self.lr_dict = OrderedDict()  # it:lr
        self.is_in_finetuning_phase = False

        self.model = kwargs["model"]
        self.run_config = kwargs["config"]
        self.callbacks = kwargs["callbacks"]

        self.optimizer = None  # To be set
        self.n_total_iterations = None

    def after_initialization(self):
        pass

    def set_optimizer(self, opt, **kwargs):
        self.optimizer = opt
        if "n_total_iterations" in kwargs:
            self.n_total_iterations = kwargs["n_total_iterations"]

    @torch.no_grad()
    def after_training_iteration(self, **kwargs):
        """Called after each training iteration"""
        if not self.is_in_finetuning_phase:
            self.lr_dict[kwargs["it"]] = kwargs["lr"]

    def at_train_begin(self):
        """Called before training begins"""
        pass

    def at_epoch_start(self, **kwargs):
        """Called before the epoch starts"""
        pass

    def at_epoch_end(self, **kwargs):
        """Called at epoch end"""
        pass

    def at_train_end(self, **kwargs):
        """Called at the end of training"""
        pass

    def final(self):
        pass

    def set_to_finetuning_phase(self):
        self.is_in_finetuning_phase = True


class QAT(Dense):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.phase = self.run_config["phase"]
        self.n_phases = self.run_config["n_phases"]
        self.n_epochs_per_phase = self.run_config["n_epochs_per_phase"]
        print("n_epochs_per_phase : ", self.n_epochs_per_phase)
        print("phase : ", self.n_phases)

    def at_train_end(self, **kwargs):
        self.callbacks["after_pruning_callback"]()
        self.finetuning_step(
            phase=self.phase,
        )

    def finetuning_step(self, phase):
        self.callbacks["finetuning_callback"](
            n_epochs_finetune=self.n_epochs_per_phase,
            phase=phase,
        )

    def final(self):
        super().final()
        self.callbacks["final_log_callback"]()
