import argparse
import os
import random

import numpy as np
import torch
import wandb

from mattersim.datasets.dataset import IceWaterDataloader
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.forcefield.potential_finetune import Potential_finetune


def main(args):
    wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
    args.run_name = (
        "finetune-n"
        + str(args.train_size)
        + "-h"
        + str(args.head_lr)
        + "-be"
        + str(args.embedding_lr)
    )
    if args.suffix is not None:
        args.run_name += "_" + args.suffix
    if args.wandb:
        wandb.init(
            project="IceWater-finetune-v0308",
            name=args.run_name,
            config=args,
        )
    args_dict = vars(args)
    args_dict.update(vars(wandb.config))
    args_dict["save_path"] = os.path.join(
        args_dict["store_path"], args_dict["run_name"]
    )
    if args.wandb:
        args_dict["wandb"] = wandb

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data, val_data, _ = IceWaterDataloader(
        root="./data", dataset_seed=args.dataset_seed, train_size=args.train_size
    ).data
    assert len(train_data) >= 100 and len(val_data) == 100
    print(f"Load {len(train_data)} for training, and {len(val_data)} for validation")

    # get atoms list
    train_energies = [x.info["TotEnergy"] for x in train_data]
    train_forces = [x.arrays["force"] for x in train_data]
    dataloader = build_dataloader(
        train_data,
        train_energies,
        train_forces,
        shuffle=True,
        pin_memory=True,
        **args_dict,
    )
    # build energy normalization module
    scale = AtomScaling(
        atoms=train_data,
        total_energy=train_energies,
        forces=train_forces,
        verbose=True,
        **args_dict,
    )

    # get atoms list
    val_energies = [x.info["TotEnergy"] for x in val_data]
    val_forces = [x.arrays["force"] for x in val_data]
    val_dataloader = build_dataloader(
        val_data, val_energies, val_forces, shuffle=True, pin_memory=True, **args_dict
    )
    potential = Potential_finetune.from_checkpoint(
        load_path="mattersim-v1.0.0-5m", device="cuda:2"
    )

    potential.train_model(
        dataloader,
        val_dataloader,
        ckpt_interval=50,
        metric_name="val_mae_f",
        **args_dict,
    )

    wandb.save(os.path.join(args_dict["save_path"], "best_model.pth"))


if __name__ == "__main__":
    # Some important arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--train_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--max_l", type=int, default=4)
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody_cutoff", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--force_loss_ratio", type=float, default=1.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    # parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--store_path", type=str, default="./results")
    # parser.add_argument("--store_path", type=str, default="/mnt/data/IceWater")
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--init_scale", type=float, default=32.17)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--multiprocessing", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--max_z", type=int, default=94)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--multithreading", type=int, default=0)
    # finetune
    parser.add_argument("--head_lr", type=float, default=2e-3)
    parser.add_argument("--backbone_lr", type=float, default=0.0)
    parser.add_argument("--embedding_lr", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--dataset_seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
