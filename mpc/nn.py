"""A class for cloning an MPC policy using a neural network"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb, time


class PolicyCloningModel(torch.nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        hidden_layer_width: int,
        n_state_dims: int,
        n_control_dims: int,
        state_space: List[Tuple[float, float]],
        load_from_file: Optional[str] = None,
    ):
        """
        A model for cloning a policy.

        args:
            hidden_layers: how many hidden layers to have
            hidden_layer_width: how many neurons per hidden layer
            n_state_dims: how many input state dimensions
            n_control_dims: how many output control dimensions
            state_space: a list of lower and upper bounds for each state dimension
            load_from_file: a path to a file containing a saved instance of a policy
                cloning model. If provided, ignores all other arguments and uses the
                saved parameters.
        """
        super(PolicyCloningModel, self).__init__()

        # If a save file is provided, use the saved parameters
        saved_data: Dict[str, Any] = {}
        if load_from_file is not None:
            saved_data = torch.load(load_from_file)
            self.hidden_layers = saved_data["hidden_layers"]
            self.hidden_layer_width = saved_data["hidden_layer_width"]
            self.n_state_dims = saved_data["n_state_dims"]
            self.n_control_dims = saved_data["n_control_dims"]
            self.state_space = saved_data["state_space"]
        else:  # otherwise, use the provided parameters
            self.hidden_layers = hidden_layers
            self.hidden_layer_width = hidden_layer_width
            self.n_state_dims = n_state_dims
            self.n_control_dims = n_control_dims
            self.state_space = state_space

        # Construct the policy network
        self.policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.policy_layers["input_linear"] = nn.Linear(
            n_state_dims,
            self.hidden_layer_width,
        )
        self.policy_layers["input_activation"] = nn.ReLU()
        for i in range(self.hidden_layers):
            # if i == 0:
            #     self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
            #         self.n_state_dims, self.hidden_layer_width
            #     )
            # else:
            self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
            self.hidden_layer_width, self.hidden_layer_width
            )
            self.policy_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.policy_layers["output_linear"] = nn.Linear(
            self.hidden_layer_width, self.n_control_dims
        )
        self.policy_nn = nn.Sequential(self.policy_layers)

        # Load the weights and biases if provided
        if load_from_file is not None:
            self.load_state_dict(saved_data["state_dict"])

    def forward(self, x: torch.Tensor):
        return self.policy_nn(x)

    def eval_np(self, x: np.ndarray):
        return self.policy_nn(torch.from_numpy(x).float()).detach().numpy()

    def save_to_file(self, save_path: str):
        save_data = {
            "hidden_layers": self.hidden_layers,
            "hidden_layer_width": self.hidden_layer_width,
            "n_state_dims": self.n_state_dims,
            "n_control_dims": self.n_control_dims,
            "state_space": self.state_space,
            "state_dict": self.state_dict(),
        }
        torch.save(save_data, save_path)
    
    def clone(
        self,
        expert: Callable[[torch.Tensor], torch.Tensor],
        n_pts: int,
        n_epochs: int,
        learning_rate: float,
        batch_size: int = 128,
        save_path: Optional[str] = None,
        saved_data_path: Optional[str] = None,
        ):
        # 0) start a run (name/project however you like)
        run = wandb.init(
            project="il_from_mpc",
            name="unicycle_bc",
            config=dict(
                n_pts=n_pts, n_epochs=n_epochs, lr=learning_rate,
                batch_size=batch_size,
                hidden_layers=self.hidden_layers,
                hidden_layer_width=self.hidden_layer_width,
            ),
            save_code=False,
        )
        t0 = time.time()

        # 1) load or generate data (your code unchanged) -> x_train, u_expert
        if saved_data_path is not None:
            x_train = torch.load(f"{saved_data_path}x_train_{n_pts-1000}.pt", weights_only=True)
            u_expert = torch.load(f"{saved_data_path}u_expert_{n_pts-1000}.pt", weights_only=True)
        else:
            x_train = torch.zeros((n_pts, self.n_state_dims))
            for dim in range(self.n_state_dims):
                x_train[:, dim] = torch.Tensor(n_pts).uniform_(*self.state_space[dim])

                # Now get the expert's control input at each of those points
                u_expert = torch.zeros((n_pts, self.n_control_dims))
                data_gen_range = tqdm(range(n_pts))
                data_gen_range.set_description("Generating training data...")
                for i in data_gen_range:
                    u_expert[i, :] = expert(x_train[i, :])
                    if i%1000 == 0:
                        print(f"Saving {i} training data points")
                        torch.save(x_train, f"x_train_{i}.pt")
                        torch.save(u_expert, f"u_expert_{i}.pt")

        # 2) tiny train/val split for progress signals (20% val)
        N = x_train.shape[0]
        n_val = max(int(0.2 * N), max(256, batch_size))
        n_tr  = N - n_val
        x_tr,  u_tr  = x_train[:n_tr],  u_expert[:n_tr]
        x_val, u_val = x_train[n_tr:],  u_expert[n_tr:]

        # 3) optimizer/loss (your code)
        mse_loss_fn = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_tr)
            loss_accum = 0.0

            for i in range(0, n_tr, batch_size):
                idx = permutation[i:i+batch_size]
                xb, ub = x_tr[idx], u_tr[idx]

                u_pred = self(xb)
                loss = mse_loss_fn(u_pred, ub)

                # L1 reg (kept)
                for layer in self.policy_nn:
                    if hasattr(layer, "weight"):
                        loss = loss + 0.001 * learning_rate * torch.norm(layer.weight, p=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += float(loss.detach())

            # 4) compute epoch metrics
            train_loss = loss_accum / (n_tr / batch_size)
            with torch.no_grad():
                self.eval()
                val_loss = float(mse_loss_fn(self(x_val), u_val))
                self.train()

            # 5) log progress (this is the whole point)
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
                "train/lr": optimizer.param_groups[0].get("lr", learning_rate),
                "meta/elapsed_s": time.time() - t0,
            }, step=epoch)

            print(f"Epoch {epoch}: train {train_loss:.6f} | val {val_loss:.6f}")

        if save_path is not None:
            self.save_to_file(save_path)
            wandb.save(save_path)   # makes the file downloadable from the run

        wandb.finish()

    # def clone(
    #     self,
    #     expert: Callable[[torch.Tensor], torch.Tensor],
    #     n_pts: int,
    #     n_epochs: int,
    #     learning_rate: float,
    #     batch_size: int = 128,
    #     save_path: Optional[str] = None,
    #     saved_data_path: Optional[str] = None,
    # ):
    #     """Clone the provided expert policy. Uses dead-simple supervised regression
    #     to clone the policy (no DAgger currently).

    #     args:
    #         expert: the policy to clone
    #         n_pts: the number of points in the cloning dataset
    #         n_epochs: the number of epochs to train for
    #         learning_rate: step size
    #         batch_size: size of mini-batches
    #         save_path: path to save the file (if none, will not save the model)
    #     """
    #     # Generate some training data
    #     # Start by sampling points uniformly from the state space
    #     if saved_data_path is not None:
    #         x_train = torch.load(f"{saved_data_path}x_train_{n_pts-1000}.pt", weights_only=True)
    #         u_expert = torch.load(f"{saved_data_path}u_expert_{n_pts-1000}.pt", weights_only=True)
    #     else:
    #         x_train = torch.zeros((n_pts, self.n_state_dims))
    #         for dim in range(self.n_state_dims):
    #             x_train[:, dim] = torch.Tensor(n_pts).uniform_(*self.state_space[dim])

    #         # Now get the expert's control input at each of those points
    #         u_expert = torch.zeros((n_pts, self.n_control_dims))
    #         data_gen_range = tqdm(range(n_pts))
    #         data_gen_range.set_description("Generating training data...")
    #         for i in data_gen_range:
    #             u_expert[i, :] = expert(x_train[i, :])
    #             if i%1000 == 0:
    #                 print(f"Saving {i} training data points")
    #                 torch.save(x_train, f"x_train_{i}.pt")
    #                 torch.save(u_expert, f"u_expert_{i}.pt")

    #     # Make a loss function and optimizer
    #     mse_loss_fn = torch.nn.MSELoss(reduction="mean")
    #     optimizer = torch.optim.Adam(
    #         self.parameters(), lr=learning_rate
    #     )

    #     # Optimize in mini-batches
    #     for epoch in range(n_epochs):
    #         permutation = torch.randperm(n_pts)

    #         loss_accumulated = 0.0
    #         epoch_range = tqdm(range(0, n_pts, batch_size))
    #         epoch_range.set_description(f"Epoch {epoch} training...")
    #         for i in epoch_range:
    #             batch_indices = permutation[i : i + batch_size]
    #             x_batch = x_train[batch_indices]
    #             u_expert_batch = u_expert[batch_indices]

    #             # Forward pass: predict the control input
    #             u_predicted = self(x_batch)

    #             # Compute the loss and backpropagate
    #             loss = mse_loss_fn(u_predicted, u_expert_batch)

    #             # Add L1 regularization
    #             for layer in self.policy_nn:
    #                 if not hasattr(layer, "weight"):
    #                     continue
    #                 loss += 0.001 * learning_rate * torch.norm(layer.weight, p=1)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             loss_accumulated += loss.detach()

    #         print(f"Epoch {epoch}: {loss_accumulated / (n_pts / batch_size)}")

    #     if save_path is not None:
    #         self.save_to_file(save_path)


