
import os
import shutil
import socket
import sys
import yaml
import tempfile
import argparse
from contextlib import contextmanager

import torch
import wandb

from runners.ensembleRunner import ensembleRunner
from runners.pretrainedRunner import pretrainedRunner
from runners.scratchRunner import scratchRunner
from runners.warmupQATRunner import warmupQATRunner

from utilities.utilities import Utilities as Utils
import os
debug = "--debug" in sys.argv
defaults = dict(
    # System
    run_id=1,  # The run id, determines the original random seed
    computer=socket.gethostname(),  # The computer that runs the experiment
    # Setup
    dataset="mnist",  # The dataset to use, see config.py for available options
    arch="Simple",  # The architecture to use, see models/ for available options
    n_epochs=2,  # The number of epochs to pretrain the model for (Note: this only controls the pretraining)
    batch_size=1028,  # The batch size to use
    # Efficiency
    use_amp=False,  # Whether to use automatic mixed precision
    # Optimizer
    optimizer="SGD",  # The optimizer to use for pretraining/retraining, currently only SGD implemented
    learning_rate="(ContinueCosine, 0.1)",  # The learning rate to use for pretraining
    n_epochs_warmup=None,  # The number of epochs to warmup the lr, must be an int
    momentum=0.9,  # The momentum to use for the optimizer
    weight_decay=0.0001,  # The weight decay to use for the optimizer
    # Sparsifying strategy
    strategy="Dense",  # The strategy to use, see strategies/ for available options. 'Dense' = pretraining, 'QAT' = Quantization aware training, 'Ensemble' = ensembl/soup methods
    # pruning_selector="global",  # Pruning allocation, must be in ['global', 'uniform', 'random']
    # Retraining
    phase=1,  # The current phase of QAT/Ensemble
    n_phases=1,  # The total number of phases to run
    n_epochs_per_phase=1,  # The number of epochs to retrain for each phase
    retrain_schedule="CLR",  # The retrain lr schedule, must be in ['FT', 'LRW', 'SLR', 'CLR', 'LLR', 'COS']
    # Ensemble method
    ensemble_method="UniformEnsembling",  # The ensemble/soup method to use, must be in ['UniformEnsembling', 'GreedySoup']
    ensemble_by="seed",  # The parameter controlling what is varied during retraining, must be in ['seed', 'weight_decay', 'retrain_length', 'retrain_schedule']
    split_val=None,  # The value to split the ensemble_by parameter on, e.g. ensemble_by='weight_decay' and split_val=0.0001 will retrain with a weight decay of 0.0001
    n_splits_total=3,  # The total number of splits we expect to have, will raise an error if more models to average found
    bn_recalibration_frac=1.0,  # The fraction of the dataset to use for recalibrating the batch norm layers, must be in [0,1]
    # Activation Quantization
    a_mode="lsq",
    a_bit=None,  # If a_bit is None -> Dense, else -> LSQ
    a_per_channel=None,
    a_symmetric=None,
    a_all_positive=None,
    # Weight Quantization
    w_mode="lsq",
    w_bit=None,  # If w_bit is None -> Dense, else -> LSQ
    w_per_channel=None,
    w_symmetric=None,
    w_all_positive=None,
    # Quantization Exception Layers
    excepts="conv1,fc,linear",
    # Pruning
    pruning_ratio=0.0,
    #
    total_epochs=0,
    n_epoch_soup_after_finetune = 0 ,
    weight_levels = None,
    act_levels = None,
    boundaryRange = None,
    update_quant = None,
    update_pruning = None,
    freeze_batchnorm_bool = None,

)

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

parser = argparse.ArgumentParser(description="QMS")
parser.add_argument("--run_id", type=int, default=1)

parser.add_argument("--cifar10_pretrained", action="store_true")
parser.add_argument("--cifar100_pretrained", action="store_true")
parser.add_argument("--imagenet_pretrained", action="store_true")
parser.add_argument("--tinyimagenet_pretrained" ,action="store_true")

parser.add_argument("--cifar10_qat", action="store_true")
parser.add_argument("--cifar100_qat", action="store_true")
parser.add_argument("--imagenet_qat", action="store_true")
parser.add_argument("--tinyimagenet_qat",action="store_true")

parser.add_argument("--cifar10_qms", action="store_true")
parser.add_argument("--cifar100_qms", action="store_true")
parser.add_argument("--imagenet_qms", action="store_true")
parser.add_argument("--tinyimagenet_qms" , action="store_true")


parser.add_argument("--arch", type=str, default="Simple")
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--phase", type=int, default=None)
parser.add_argument("--n_phases", type=int, default=None)
parser.add_argument("--n_epochs_per_phase", type=int, default=None)
parser.add_argument("--n_splits_total", type=int, default=None)
parser.add_argument("--split_val", type=int, default=None)

parser.add_argument("--retrain_schedule", type=str, default=None)
parser.add_argument("--strategy", type=str, required=True)

parser.add_argument("--ensemble_method", type=str, default=None)
parser.add_argument("--ensemble_by", type=str, default=None)
parser.add_argument("--pruning_ratio", type=float, default=0.0)
parser.add_argument("--boundaryRange",  type=float, default=0.0)


parser.add_argument('--w_bit', type=int, default=0, help='bit-width for weights')
parser.add_argument('--a_bit', type=int, default=0, help='bit-width for activations')
parser.add_argument('--weight_levels', type=int, default=0, help='number of quantization levels for weights')
parser.add_argument('--act_levels', type=int, default=0, help='number of quantization levels for activations')

parser.add_argument(
    "--total_epochs", type=int, help="Num of total epoch (For learning rate schedule)"
)

parser.add_argument("--n_epoch_soup_after_finetune", type=int, default=None)
parser.add_argument("--update_quant" , type=str, default=None)
parser.add_argument("--update_pruning" , type=str, default=None)
parser.add_argument("--freeze_batchnorm_bool", type=str, default=None)

cfg = parser.parse_args()


if not debug:
    if cfg.cifar10_pretrained:
        with open("configs/cifar10_pretrained.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.cifar100_pretrained:
        with open("configs/cifar100_pretrained.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.imagenet_pretrained:

        with open("configs/imagenet_pretrained.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)
        
    elif cfg.tinyimagenet_pretrained:

        with open("configs/tinyimagenet_pretrained.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)
    elif cfg.tinyimagenet_qat:

        with open("configs/tinyimagenet_qat.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)
    elif cfg.tinyimagenet_qms:

        with open("configs/tinyimagenet_qms.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)
        
    elif cfg.cifar10_qat:
        with open("configs/cifar10_qat.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.cifar10_qms:
        with open("configs/cifar10_qms.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.cifar100_qat:
        with open("configs/cifar100_qat.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.cifar100_qms:
        with open("configs/cifar100_qms.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.imagenet_qat:
        with open("configs/imagenet_qat.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

    elif cfg.imagenet_qms:
        with open("configs/imagenet_qms.yaml", "r") as f:
            cifar_config = yaml.safe_load(f)
        defaults.update(cifar_config)

defaults.update(
    {
        "run_id": cfg.run_id,
        "computer": socket.gethostname(),
        "arch": cfg.arch,
        "n_epochs": cfg.n_epochs,
        "dataset": cfg.dataset,
        "batch_size": cfg.batch_size,
        "phase": cfg.phase,
        "n_phases": cfg.n_phases,
        "n_epochs_per_phase": cfg.n_epochs_per_phase,
        "n_splits_total": cfg.n_splits_total,
        "split_val": cfg.split_val,
        "retrain_schedule": cfg.retrain_schedule,
        "strategy": cfg.strategy,
        "ensemble_method": cfg.ensemble_method,
        "ensemble_by": cfg.ensemble_by,
        "pruning_ratio": cfg.pruning_ratio,
        "total_epochs": cfg.total_epochs,
        "n_epoch_soup_after_finetune": cfg.n_epoch_soup_after_finetune,
        "w_bit" : cfg.w_bit,
        "a_bit" : cfg.a_bit,
        "weight_levels" : cfg.weight_levels,
        "act_levels" : cfg.act_levels,
        "boundaryRange" : cfg.boundaryRange,
        "update_pruning": cfg.update_pruning,
        "update_quant": cfg.update_quant,
        "freeze_batchnorm_bool": cfg.freeze_batchnorm_bool,
        
    }
)

print(defaults['w_bit'])


if cfg.cifar10_pretrained or cfg.cifar100_pretrained or cfg.imagenet_pretrained or cfg.tinyimagenet_pretrained :
    wandb.init(
        config=defaults,
        project=f"{cfg.dataset}_{cfg.arch}_dense",
        entity=None,
    )
    wandb.run.name = f"pretrained_id{defaults['run_id']}"
elif cfg.cifar10_qms or cfg.cifar100_qms or cfg.imagenet_qms or cfg.tinyimagenet_qms:
    wandb.init(
        config=defaults,
        project=f"{defaults['dataset']}_{defaults['arch']}_W{defaults['w_bit']}A{defaults['a_bit']}_P{defaults['pruning_ratio']}_qms_continue_23FreezeQ",
        entity=None,
    )
    wandb.run.name = f"qms_id{defaults['run_id']}_phase{defaults['phase']}"

elif cfg.cifar10_qat or cfg.cifar100_qat or cfg.imagenet_qat or cfg.tinyimagenet_qat:
    wandb.init(
        config=defaults,
        project=f"{defaults['dataset']}_{defaults['arch']}_W{defaults['w_bit']}A{defaults['a_bit']}_Epoch{defaults['n_epochs']}_WarmupQAT_continue",
        entity=None,
    )
    wandb.run.name = f"warmup_qat_id{defaults['run_id']}"

config = wandb.config
config = Utils.update_config_with_default(config, defaults)
ngpus = torch.cuda.device_count()
if ngpus > 0:
    config.update(dict(device="cuda:0"), allow_val_change=True)
else:
    config.update(dict(device="cpu"), allow_val_change=True)

# Set log_dir based on wandb run directory
config.update({"log_dir": wandb.run.dir}, allow_val_change=True)


@contextmanager
def tempdir():
    tmp_root = "/scratch/local/"
    tmp_path = os.path.join(tmp_root, "tmp")
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write("Failed to clean up temp dir {}".format(path))


with tempdir() as tmp_dir:
    # At the moment, QAT is the only strategy that requires a pretrained model, all others start from scratch
    config.update({"tmp_dir": tmp_dir})
    if config.strategy == "Ensemble":
        runner = ensembleRunner(config=config)
    elif config.strategy in "QAT":
        # Use the pretrainedRunner
        runner = pretrainedRunner(config=config)
    elif config.strategy in "WarmupQAT":
        # Use the pretrainedRunner
        runner = warmupQATRunner(config=config)
    elif config.strategy == "Dense":
        # Use the scratchRunner
        runner = scratchRunner(config=config)
    else:
        raise NotImplementedError(f"Strategy {config.strategy} not implemented.")
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
