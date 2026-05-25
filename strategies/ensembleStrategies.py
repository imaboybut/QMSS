
import sys
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch

from strategies import strategies as usual_strategies
from utilities.utilities import Candidate
from utilities.utilities import Utilities as Utils


class EnsemblingBaseClass(usual_strategies.Dense):
    """Ensembling Base Class"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.candidate_model_list = kwargs["candidate_models"]
        self.runner = kwargs["runner"]
        self.selected_models = None
        self.soup_metrics = {soup_type: {} for soup_type in ["candidates", "selected"]} 
        

    @torch.no_grad()

    def get_soup_metrics(self, soup_list: list[Candidate]):

        # Load the models
        model_list = [candidate.get_model_weights() for candidate in soup_list]

        soup_metrics = {
            "max_barycentre_distance": Utils.get_barycentre_l2_distance(model_list),
            "min_barycentre_distance": Utils.get_barycentre_l2_distance(
                model_list, maximize=False
            ),
        }

        for metric_name, metric_fn in zip(
            ["l2_distance", "angle"], [Utils.get_l2_distance, Utils.get_angle]
        ):
            for agg_name, agg_fn in zip(
                ["max", "min", "mean"], [torch.max, torch.min, torch.mean]
            ):
                soup_metrics[f"{agg_name}_{metric_name}"] = (
                    Utils.aggregate_group_metrics(
                        models=model_list, metric_fn=metric_fn, aggregate_fn=agg_fn
                    )
                )
        return soup_metrics



    def create_ensemble(self, **kwargs):
        n_models = len(self.candidate_model_list)
        assert n_models >= 2, "Not enough models to ensemble"

    @torch.no_grad()
    def average_models(
        self,
        soup_list: list[Candidate],
        soup_weights: torch.Tensor = None,
        device: torch.device = torch.device("cpu"),
    ):
        if soup_weights is None:
            soup_weights = torch.ones(len(soup_list)) / len(soup_list)
        ensemble_state_dict = OrderedDict()

        for idx, candidate in enumerate(soup_list):
            candidate_id, candidate_file = candidate.id, candidate.file
            state_dict = torch.load(
                candidate_file, map_location=torch.device("cpu")
            )  # Load to CPU to avoid memory overhead
            for key, val in state_dict.items():
                factor = soup_weights[idx].item()  # No need to use tensor here
                if "_mask" in key:
                    # We dont want to average the masks, hence we skip them and add later
                    continue
                if key not in ensemble_state_dict:
                    ensemble_state_dict[key] = (
                        factor * val.detach().clone()
                    )  # Important: clone otherwise we modify the tensors
                else:
                    ensemble_state_dict[key] += (
                        factor * val.detach().clone()
                    )  # Important: clone otherwise we modify the tensors

        # Add the masks from the last state_dict
        for key, val in state_dict.items():
            if "_mask" in key:
                ensemble_state_dict[key] = val.detach().clone()

        del state_dict
        del factor
        del soup_weights

        return ensemble_state_dict

    def final(self):
        self.callbacks["final_log_callback"]()

    def get_ensemble_metrics(self):
        if self.selected_models == "all": 
            # We have already collected the metrics for all models
            if self.soup_metrics["candidates"] is None:
                self.soup_metrics["selected"] = 0
            else:
                self.soup_metrics["selected"] = self.soup_metrics["candidates"]

        return self.soup_metrics


class UniformEnsembling(EnsemblingBaseClass):
    """Just averages all models"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def create_ensemble(self, **kwargs):
        super().create_ensemble(**kwargs)

        device = torch.device("cpu")
        soup_weights = self.get_soup_weights(soup_list=self.candidate_model_list)
        ensemble_state_dict = self.average_models(
            soup_list=self.candidate_model_list,
            soup_weights=soup_weights,
            device=device,
        )
        self.selected_models = "all"
        return ensemble_state_dict

    def get_soup_weights(self, soup_list: list[Candidate]): 
        uniform_factor = 1.0 / len(soup_list)
        return torch.tensor([uniform_factor] * len(soup_list))

