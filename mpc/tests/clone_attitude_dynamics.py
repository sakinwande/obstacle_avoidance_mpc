"""Test the obstacle avoidance MPC for a quadrotor"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from NNet.converters.onnx2nnet import onnx2nnet

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
    squared_error_terminal_cost,
)
from mpc.dynamics_constraints import attitude_control_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf2
from mpc.simulator import simulate_nn
from mpc.network_utils import pytorch_to_nnet

from mpc.nn import PolicyCloningModel



n_states = 6
n_controls = 3
horizon = 30
dt = 0.1
dynamics_fn = attitude_control_dynamics 
radius = 0.5
margin = 0.05
#Define the center of the sphere to be far from op region
center = [-0.65, 0.75, -0.3, -0.1, -0.45, 0.1]
radii = [0.1, 0.1, 0.15, 0.15, 0.1, 0.15]
#Define the indices of the state that are in the sphere
indices = [i for i in range(6)]
state_space = [
    (-3.0, 3.0),  # px
    (-3.0, 3.0),  # py
    (-3.0, 3.0),  # pz
    (-1.0, 1.0),  # vx
    (-1.0, 1.0),  # vy
    (-1.0, 1.0),  # vz
]



def define_mpc_expert():
    # -------------------------------------------
    # Define the MPC problem
    # -------------------------------------------

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf2(x, radius, indices, center), margin)]
    # Define costs to make the quad go up
    x_goal = np.zeros(6)
    x_goal[3] = 0.1
    x_goal[5] = -0.1

    running_costs = 0.1 * np.eye(6)
    control_costs = 0.1 * np.eye(3)
    terminal_costs = 0.1*np.eye(6) 

    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * running_costs, control_costs)
    
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal,dt * terminal_costs)

    # Define control bounds
    control_bounds = [1.0, 1.0, 1.0]

    # Define MPC problem
    opti, x0_variables, u0_variables, x_variables, u_variables = construct_MPC_problem(
        n_states,
        n_controls,
        horizon,
        dt,
        dynamics_fn,
        obstacle_fns,
        running_cost_fn,
        terminal_cost_fn,
        control_bounds,
    )

    # Wrap the MPC problem to accept a tensor and return a tensor
    max_tries = 10

    def mpc_expert(current_state: torch.Tensor) -> torch.Tensor:
        tries = 0
        success = False
        x_guess = None
        u_guess = None
        while not success and tries < max_tries:
            success, control_output, _, _ = solve_MPC_problem(
                opti.copy(),
                x0_variables,
                u0_variables,
                current_state.detach().numpy(),
                x_variables=x_variables,
                u_variables=u_variables,
                x_guess=x_guess,
                u_guess=u_guess,
            )
            tries += 1

        if not success:
            print(f"failed after {tries} tries")

        return torch.from_numpy(control_output)

    return mpc_expert


def clone_mpc(train=True):
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_mpc_expert()
    hidden_layers = 6
    hidden_layer_width = 128
    cloned_policy = PolicyCloningModel(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space,
        #load_from_file="mpc/tests/data/cloned_quad_policy_weight_decay.pth",
    )

    n_pts = int(5e4)
    n_epochs = 5000
    learning_rate = 1e-3
    if train:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path="mpc/tests/data/cloned_att_policy_weight_decay.pth",
            saved_data_path=None,
        )

    return cloned_policy


def   simulate_and_plot(policy):
    # -------------------------------------------
    # Plot a rollout of the cloned
    # -------------------------------------------
    ngrids = 5
    x1s = np.linspace(-0.4, 0.4, ngrids)
    x2s = np.linspace(-0.4, 0.4, ngrids)
    x3s = np.linspace(-0.4, 0.4, ngrids)
    x4s = np.linspace(-0.4, 0.4, ngrids)
    x5s = np.linspace(-0.4, 0.4, ngrids)
    x6s = np.linspace(-0.4, 0.4, ngrids)
    x0s = []
    for i6 in x6s:
        for i5 in x5s:
            for i4 in x4s:
                for i3 in x3s:
                    for i2 in x2s:
                        for i1 in x1s:
                            x0s.append([i1, i2, i3, i4, i5, i6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    n_steps = 50
    for x0 in x0s:
        _, x, u = simulate_nn(
            policy,
            x0,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
        )

        # Plot it (in x-y plane)
        ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        ax_xz.plot(x0[0], x0[2], "ro")
        ax_xz.plot(x[:, 0], x[:, 2], "r-", linewidth=1)

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    obs_z = radius * np.sin(theta) + center[2]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    margin_z = (radius + margin) * np.sin(theta) + center[2]
    ax_xy.plot(obs_x, obs_y, "k-")
    ax_xy.plot(margin_x, margin_y, "k:")
    ax_xz.plot(obs_x, obs_z, "k-", label="Obstacle")
    ax_xz.plot(margin_x, margin_z, "k:", label="Safety margin")

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-5.0, 5.0])
    ax_xy.set_ylim([-5.0, 5.0])
    ax_xz.set_xlim([-5.0, 5.0])
    ax_xz.set_ylim([-5.0, 5.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.show()


def save_to_onnx(policy):
    """Save to an onnx file"""
    save_path = "mpc/tests/data/cloned_quad_policy_weight_decay.onnx"
    pytorch_to_nnet(policy, n_states, n_controls, save_path)

    input_mins = [state_range[0] for state_range in state_space]
    input_maxes = [state_range[1] for state_range in state_space]
    means = [0.5 * (state_range[0] + state_range[1]) for state_range in state_space]
    means += [0.0]
    ranges = [state_range[1] - state_range[0] for state_range in state_space]
    ranges += [1.0]
    onnx2nnet(save_path, input_mins, input_maxes, means, ranges)

if __name__ == "__main__":
    policy = clone_mpc(train=True)
    save_to_onnx(policy)
    simulate_and_plot(policy)
    boo = 1
