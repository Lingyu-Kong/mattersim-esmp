"""
Potential
"""

import os
import random
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_ema import ExponentialMovingAverage
from torchmetrics import MeanMetric

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.jit_compile_tools.jit import compile_mode
from mattersim.utils.download_utils import download_checkpoint


def build_optimizer_and_scheduler(
    model,
    head_lr: float = 2e-3,
    backbone_lr: float = 0,
    embedding_lr: float = 0,
    **kwargs,
):
    """
    Build optimizer and scheduler for fine-tuning
    """
    backbone_params = []
    head_params = []
    embedding_params = []
    # edge_encoder, graph_conv, final, atom_embedding
    for name, param in model.named_parameters():
        if "graph_conv" in name:
            backbone_params.append(param)
            # print(f'Backbone: {name}')
        elif "atom_embedding" in name or "edge_encoder" in name:
            embedding_params.append(param)
            # print(f'Embedding: {name}')
        elif "final" in name:
            head_params.append(param)
            # print(f'Head: {name}')
        else:
            raise ValueError(f"Unknown parameter: {name}")
            # print(f'Unassigned: {name}')
    step_size = kwargs.get("step_size", 10)
    gamma = kwargs.get("gamma", 0.95)
    optimizer_list = [
        Adam(backbone_params, lr=backbone_lr, eps=1e-7),
        Adam(head_params, lr=head_lr, eps=1e-7),
        Adam(embedding_params, lr=embedding_lr, eps=1e-7),
    ]
    scheduler_list = [
        StepLR(optimizer_list[0], step_size=step_size, gamma=gamma),
        StepLR(optimizer_list[1], step_size=step_size, gamma=gamma),
        StepLR(optimizer_list[2], step_size=step_size, gamma=gamma),
    ]
    return optimizer_list, scheduler_list


@compile_mode("script")
class Potential_finetune(nn.Module):
    """
    A wrapper class for fine-tuning M3GNet
    """

    def __init__(
        self,
        model,
        optimizer_list: list | None = None,
        scheduler_list: list | None = None,
        ema=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allow_tf32=False,
        **kwargs,
    ):
        """
        Args:
            potential : a force field model
            lr : learning rate
            scheduler : a torch scheduler
            normalizer : an energy normalization module
        """
        super().__init__()
        self.model = model
        if optimizer_list is None and scheduler_list is None:
            self.optimizer_list, self.scheduler_list = build_optimizer_and_scheduler(
                self.model, **kwargs
            )
        else:
            self.optimizer_list = optimizer_list
            self.scheduler_list = scheduler_list
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        self.device = device
        self.to(device)
        if ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=kwargs.get("ema_decay", 0.99)
            )
        else:
            self.ema = ema
        self.model_name = kwargs.get("model_name", "m3gnet")
        self.validation_metrics = kwargs.get("validation_metrics", {"loss": 10000.0})
        self.last_epoch = kwargs.get("last_epoch", -1)
        self.description = kwargs.get("description", "")
        print("Potential_finetune class is specifically designed for fine-tuning.")

    def train_model(
        self,
        dataloader,
        val_dataloader,
        loss=torch.nn.MSELoss(),
        include_forces: bool = False,
        include_stresses: bool = False,
        force_loss_ratio: float = 1.0,
        stress_loss_ratio: float = 0.1,
        epochs: int = 100,
        early_stop_patience: int = 100,
        metric_name: str = "val_loss",
        wandb=None,
        save_checkpoint: bool = False,
        save_path: str = "./results/",
        ckpt_interval: int = 10,
        multi_head: bool = False,
        dataset_name_list: List[str] | None = None,
        **kwargs,
    ):
        """
        Train model
        Args:
            dataloader: training data loader
            val_dataloader: validation data loader
            loss (torch.nn.modules.loss): loss object
            include_forces (bool) : whether to use forces as optimization targets
            include_stresses (bool) : whether to use stresses as optimization targets
            force_loss_ratio (float): the ratio of forces in loss
            stress_loss_ratio (float): the ratio of stress in loss
            ckpt_interval (int): the interval to save checkpoints
            early_stop_patience (int): the patience for early stopping
            metric_name (str): the metric used for saving `best` checkpoints and early stopping
                               supported metrics: `val_loss`, `val_mae_e`, `val_mae_f`, `val_mae_s`
        """
        for epoch in range(self.last_epoch + 1, epochs):
            print(f"Epoch: {epoch} / {epochs}")
            if not multi_head:
                self.train_one_epoch(
                    dataloader,
                    epoch,
                    loss,
                    include_forces,
                    include_stresses,
                    force_loss_ratio,
                    stress_loss_ratio,
                    wandb,
                    mode="train",
                    **kwargs,
                )
                metric = self.train_one_epoch(
                    val_dataloader,
                    epoch,
                    loss,
                    include_forces,
                    include_stresses,
                    force_loss_ratio,
                    stress_loss_ratio,
                    wandb,
                    mode="val",
                    **kwargs,
                )
            else:
                assert NotImplementedError

            if self.scheduler_list is not None:
                for scheduler in self.scheduler_list:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(metric)
                    else:
                        scheduler.step()

            self.last_epoch = epoch

            self.validation_metrics = {
                "loss": metric[0],
                "MAE_energy": metric[1],
                "MAE_force": metric[2],
                "MAE_stress": metric[3],
            }

            with self.ema.average_parameters():
                try:
                    best_model = torch.load(os.path.join(save_path, "best_model.pth"))
                    assert metric_name in [
                        "val_loss",
                        "val_mae_e",
                        "val_mae_f",
                        "val_mae_s",
                    ], (
                        f"`{metric_name}` metric name not supported."
                        " supported metrics: `val_loss`, `val_mae_e`, `val_mae_f`, `val_mae_s`"
                    )
                    saved_name = ["loss", "MAE_energy", "MAE_force", "MAE_stress"]
                    idx = ["val_loss", "val_mae_e", "val_mae_f", "val_mae_s"].index(
                        metric_name
                    )
                    if (
                        save_checkpoint is True
                        and metric[idx]
                        < best_model["validation_metrics"][saved_name[idx]]
                    ):
                        self.save(os.path.join(save_path, "best_model.pth"))
                    if epoch > best_model["last_epoch"] + early_stop_patience:
                        print("Early stopping")
                        break
                    del best_model
                except:
                    if save_checkpoint is True:
                        self.save(os.path.join(save_path, "best_model.pth"))

                if save_checkpoint is True and epoch % ckpt_interval == 0:
                    self.save(os.path.join(save_path, f"ckpt_{epoch}.pth"))
                if save_checkpoint is True:
                    self.save(os.path.join(save_path, f"last_model.pth"))

    def test_model(
        self,
        val_dataloader,
        loss: torch.nn.modules.loss = torch.nn.MSELoss(),
        include_forces: bool = False,
        include_stresses: bool = False,
        wandb=None,
        multi_head: bool = False,
        **kwargs,
    ):
        """
        Test model performance on a given dataset
        """
        if not multi_head:
            return self.train_one_epoch(
                val_dataloader,
                1,
                loss,
                include_forces,
                include_stresses,
                1.0,
                0.1,
                wandb=wandb,
                mode="val",
            )
        else:
            assert NotImplementedError

    def train_one_epoch(
        self,
        dataloader,
        epoch,
        loss,
        include_forces,
        include_stresses,
        loss_f,
        loss_s,
        wandb,
        mode="train",
        **kwargs,
    ):
        start_time = time.time()
        loss_avg = MeanMetric().to(self.device)
        train_e_mae = MeanMetric().to(self.device)
        train_f_mae = MeanMetric().to(self.device)
        train_s_mae = MeanMetric().to(self.device)

        # scaler = torch.cuda.amp.GradScaler()

        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()

        for batch_idx, graph_batch in enumerate(dataloader):
            graph_batch.to(self.device)
            input = batch_to_dict(graph_batch)
            if mode == "train":
                result = self.forward(
                    input,
                    include_forces=include_forces,
                    include_stresses=include_stresses,
                )
            elif mode == "val":
                with self.ema.average_parameters():
                    result = self.forward(
                        input,
                        include_forces=include_forces,
                        include_stresses=include_stresses,
                    )

            loss_, e_mae, f_mae, s_mae = self.loss_calc(
                graph_batch,
                result,
                loss,
                include_forces,
                include_stresses,
                loss_f,
                loss_s,
            )

            # loss backward
            if mode == "train":
                for optimizer in self.optimizer_list:
                    # TODO: Test if it works
                    optimizer.zero_grad()
                loss_.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, norm_type=2)
                # selected_params_name = ["edge_encoder.mlp.0.linear.weight", "graph_conv.0.gated_mlp_atom.g.0.linear.weight", "final.g.0.linear.weight"]
                # original_params_value = []
                # for name, param in self.model.named_parameters():
                #     if name in selected_params_name:
                #         original_params_value.append(param.clone())
                for optimizer in self.optimizer_list:
                    optimizer.step()
                # if batch_idx % 100 == 0:
                #     for i, name in enumerate(selected_params_name):
                #         for name2, param in self.model.named_parameters():
                #             if name == name2:
                #                 print(f"batch_idx {batch_idx}, {name} param diff: ", torch.sum(torch.abs(param - original_params_value[i])))
                # scaler.scale(loss_).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                self.ema.update()

            loss_avg.update(loss_.detach())
            train_e_mae.update(e_mae.detach())
            if include_forces == True:
                train_f_mae.update(f_mae.detach())
            if include_stresses == True:
                train_s_mae.update(s_mae.detach())

        loss_avg_ = loss_avg.compute().item()
        e_mae = train_e_mae.compute().item()
        if include_forces == True:
            f_mae = train_f_mae.compute().item()
        else:
            f_mae = 0
        if include_stresses == True:
            s_mae = train_s_mae.compute().item()
        else:
            s_mae = 0

        print(
            "\r%s: Loss: %.4f, MAE(e): %.4f, MAE(f): %.4f, MAE(s): %.4f, Time: %.2fs"
            % (
                mode,
                loss_avg.compute().item(),
                e_mae,
                f_mae,
                s_mae,
                time.time() - start_time,
            )
        )

        if wandb:
            wandb.log(
                {
                    f"{mode}/loss": loss_avg_,
                    f"{mode}/mae_e": e_mae,
                    f"{mode}/mae_f": f_mae,
                    f"{mode}/mae_s": s_mae,
                    f"{mode}/mae_tot": e_mae + f_mae + s_mae,
                },
                step=epoch,
            )

        if mode == "val":
            return (loss_avg_, e_mae, f_mae, s_mae)

    def loss_calc(
        self,
        graph_batch,
        result,
        loss,
        include_forces,
        include_stresses,
        loss_f=1.0,
        loss_s=0.1,
    ):
        f_mae = 0.0
        s_mae = 0.0
        e_gt = graph_batch.energy / graph_batch.num_atoms
        e_pred = result["energies"] / graph_batch.num_atoms
        loss_ = loss(e_pred, e_gt)
        e_mae = torch.nn.L1Loss()(e_pred, e_gt)
        if include_forces == True:
            f_gt = graph_batch.forces
            f_pred = result["forces"]
            loss_ += loss(f_pred, f_gt) * loss_f
            f_mae = torch.nn.L1Loss()(f_pred, f_gt)
            # f_mae = torch.mean(torch.abs(f_pred - f_gt)).item()
        if include_stresses == True:
            s_gt = graph_batch.stress
            s_pred = result["stresses"]
            loss_ += loss(s_pred, s_gt) * loss_s
            s_mae = torch.nn.L1Loss()(s_pred, s_gt)
            # s_mae = torch.mean(torch.abs((s_pred - s_gt))).item()
        return loss_, e_mae, f_mae, s_mae

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        include_forces: bool = True,
        include_stresses: bool = True,
        dataset_idx: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        get energy, force and stress from a list of graph
        Args:
            input: a dictionary contains all necessary info.
                   The `batch_to_dict` method could convert a graph_batch from pyg dataloader to the input dictionary.
            include_forces (bool): whether to include force
            include_stresses (bool): whether to include stress
            dataset_idx (int): used for multi-head model, set to -1 by default
        Returns:
            results: a dictionary, which consists of energies, forces and stresses
        """
        output = {}
        strain = torch.zeros_like(input["cell"], device=self.device)
        volume = torch.linalg.det(input["cell"])
        if include_forces is True:
            input["atom_pos"].requires_grad_(True)
        if include_stresses is True:
            strain.requires_grad_(True)
            input["cell"] = torch.matmul(
                input["cell"], (torch.eye(3, device=self.device)[None, ...] + strain)
            )
            strain_augment = torch.repeat_interleave(strain, input["num_atoms"], dim=0)
            input["atom_pos"] = torch.einsum(
                "bi, bij -> bj",
                input["atom_pos"],
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(input["cell"])

        energies = self.model.forward(input, dataset_idx)
        output["energies"] = energies
        if include_forces is True:
            forces = gradient(
                outputs=energies,
                inputs=input["atom_pos"],
                retain_graph=include_stresses or self.model.training,
                create_graph=self.model.training,
            )
            if forces is not None:
                forces = torch.neg(forces)
                output["forces"] = forces

        if include_stresses is True:
            # eV/A^3 to GPa
            grad = gradient(
                outputs=energies, inputs=strain, create_graph=self.model.training
            )
            if grad is not None:
                stresses = 1 / volume[:, None, None] * grad * 160.21766208
                output["stresses"] = stresses

        return output

    def save(self, save_path):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        checkpoint = {
            "model_name": self.model_name,
            "model": self.model.state_dict(),
            "model_args": self.model.get_model_args(),
            "optimizer_list": [
                optimizer.state_dict() for optimizer in self.optimizer_list
            ],
            "ema": self.ema.state_dict(),
            "scheduler_list": [
                scheduler.state_dict() for scheduler in self.scheduler_list
            ],
            "last_epoch": self.last_epoch,
            "validation_metrics": self.validation_metrics,
            "description": self.description,
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load(
        model_name: str = "m3gnet",
        load_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        if load_path is None:
            if model_name == "m3gnet":
                print("Loading the pre-trained M3GNet model")
                current_dir = os.path.dirname(__file__)
                load_path = os.path.join(
                    current_dir, "m3gnet/pretrained/mpf/best_model.pth"
                )
            else:
                raise NotImplementedError
        else:
            print("Loading the model from %s" % load_path)
        checkpoint = torch.load(load_path, map_location=device)
        assert checkpoint["model_name"] == model_name
        if model_name == "m3gnet":
            model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
        elif model_name == "m3gnet_multi_head":
            raise NotImplementedError
        else:
            raise NotImplementedError
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer_list, scheduler_list = build_optimizer_and_scheduler(model, **kwargs)
        for idx, optimizer in enumerate(optimizer_list):
            optimizer.load_state_dict(checkpoint["optimizer_list"][idx])
        for idx, scheduler in enumerate(scheduler_list):
            scheduler.load_state_dict(checkpoint["scheduler_list"][idx])
        last_epoch = checkpoint["last_epoch"]
        validation_metrics = checkpoint["validation_metrics"]
        description = checkpoint["description"]
        try:
            ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
            ema.load_state_dict(checkpoint["ema"])
        except:
            ema = None
        model.eval()

        del checkpoint

        return Potential_finetune(
            model,
            optimizer_list=optimizer_list,
            ema=ema,
            scheduler_list=scheduler_list,
            device=device,
            model_name=model_name,
            last_epoch=last_epoch,
            validation_metrics=validation_metrics,
            description=description,
            **kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        load_path: str = None,
        *,
        model_name: str = "m3gnet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_training_state: bool = True,
        **kwargs,
    ):
        if model_name.lower() != "m3gnet":
            raise NotImplementedError

        checkpoint_folder = os.path.expanduser("~/.local/mattersim/pretrained_models")
        os.makedirs(checkpoint_folder, exist_ok=True)
        if (
            load_path is None
            or load_path.lower() == "mattersim-v1.0.0-1m.pth"
            or load_path.lower() == "mattersim-v1.0.0-1m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-1M.pth")
            if not os.path.exists(load_path):
                download_checkpoint(
                    "mattersim-v1.0.0-1M.pth", save_folder=checkpoint_folder
                )
        elif (
            load_path.lower() == "mattersim-v1.0.0-5m.pth"
            or load_path.lower() == "mattersim-v1.0.0-5m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-5M.pth")
            if not os.path.exists(load_path):
                download_checkpoint(
                    "mattersim-v1.0.0-5M.pth", save_folder=checkpoint_folder
                )
        else:
            pass
        assert os.path.exists(load_path), f"Model file {load_path} not found"

        checkpoint = torch.load(load_path, map_location=device)

        assert checkpoint["model_name"] == model_name
        model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=False)

        if load_training_state:
            optimizer = Adam(model.parameters())
            scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except BaseException:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
                except BaseException:
                    optimizer = None
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
            except BaseException:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"].state_dict())
                except BaseException:
                    scheduler = "StepLR"
            try:
                last_epoch = checkpoint["last_epoch"]
                validation_metrics = checkpoint["validation_metrics"]
                description = checkpoint["description"]
            except BaseException:
                last_epoch = -1
                validation_metrics = {"loss": 0.0}
                description = ""
            try:
                ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
                ema.load_state_dict(checkpoint["ema"])
            except BaseException:
                ema = None
        else:
            optimizer = None
            scheduler = "StepLR"
            last_epoch = -1
            validation_metrics = {"loss": 0.0}
            description = ""
            ema = None

        model.eval()

        del checkpoint

        return cls(
            model,
            optimizer=optimizer,
            ema=ema,
            scheduler=scheduler,
            device=device,
            model_name=model_name,
            last_epoch=last_epoch,
            validation_metrics=validation_metrics,
            description=description,
            **kwargs,
        )


@torch.jit.script
def gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    retain_graph: bool = None,
    create_graph: bool = False,
) -> Optional[torch.Tensor]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(outputs)]
    grad = torch.autograd.grad(
        outputs=[
            outputs,
        ],
        inputs=[inputs],
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    if grad is not None:
        grad = grad[0]
    return grad


def batch_to_dict(graph_batch):
    atom_pos = graph_batch.atom_pos
    cell = graph_batch.cell
    pbc_offsets = graph_batch.pbc_offsets
    atom_attr = graph_batch.atom_attr
    edge_index = graph_batch.edge_index
    three_body_indices = graph_batch.three_body_indices
    num_three_body = graph_batch.num_three_body
    num_bonds = graph_batch.num_bonds
    num_triple_ij = graph_batch.num_triple_ij
    num_atoms = graph_batch.num_atoms
    num_graphs = graph_batch.num_graphs
    num_graphs = torch.tensor(num_graphs)
    batch = graph_batch.batch

    # Resemble input dictionary
    input = {}
    input["atom_pos"] = atom_pos
    input["cell"] = cell
    input["pbc_offsets"] = pbc_offsets
    input["atom_attr"] = atom_attr
    input["edge_index"] = edge_index
    input["three_body_indices"] = three_body_indices
    input["num_three_body"] = num_three_body
    input["num_bonds"] = num_bonds
    input["num_triple_ij"] = num_triple_ij
    input["num_atoms"] = num_atoms
    input["num_graphs"] = num_graphs
    input["batch"] = batch

    return input
