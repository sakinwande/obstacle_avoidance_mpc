"""Test the obstacle avoidance MPC for a quadrotor"""
import numpy as np
import torch
import matplotlib
matplotlib.use("WebAgg")   # or "TkAgg"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D plots

from NNet.converters.onnx2nnet import onnx2nnet

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
    squared_error_terminal_cost,
)
from mpc.dynamics_constraints import unicycle_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf2
from mpc.simulator import simulate_nn
from mpc.network_utils import pytorch_to_nnet

from mpc.nn import PolicyCloningModel



n_states = 4
n_controls = 2
horizon = 50
dt = 0.1
dynamics_fn = unicycle_dynamics 
radius = 0.5
margin = 0.05
#Define the center of the sphere to be far from op region
center = [0,0,0,0]
radii = [0.2, 0.2, 6.28, 4]
#Define the indices of the state that are in the sphere
indices = [1,2]
state_space = [
    (-12, 12),  # px
    (-12, 12),  # py
    (-np.pi, np.pi),  # pz
    (-1, 1),  # vx
]



def define_mpc_expert():
    # -------------------------------------------
    # Define the MPC problem
    # -------------------------------------------

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf2(x, radius, indices, center), margin)]
    # Define costs to make the quad go up
    x_goal = np.zeros(4)
    x_goal[0] = 2
    x_goal[1] = 2
    x_goal[2] = 0
    x_goal[3] = 0

    running_costs = 0.1 * np.eye(4)
    control_costs = 0.1 * np.eye(2)
    terminal_costs = 0.1*np.eye(4) 

    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * running_costs, control_costs)
    
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal,dt * terminal_costs)

    # Define control bounds
    control_bounds = [1.0, 1.0]

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

def clone_mpc(train=False):
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_mpc_expert()
    hidden_layers = 0
    hidden_layer_width = 250
    cloned_policy = PolicyCloningModel(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space,
        #load_from_file="mpc/tests/data/cloned_uni_policy_weight_decay.pth",
    )

    n_pts = int(8e4)
    n_epochs = 10000
    learning_rate = 1e-3
    if train:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path="mpc/tests/data/cloned_uni_policy_weight_decay.pth",
            saved_data_path="Training_Data/unicycle_expert_"
        )

    return cloned_policy

def simulate_and_plot(policy, *,
                      n_steps=50,
                      ngrids=5,
                      omega_limits=(-4.0, 4.0),
                      z_limits=(-2.0, 2.0),
                      xy_limits=(-15.0, 15.0),
                      out_path="rollout_xyz_xyw",
                      save_formats=("svg", "pdf", "png"),
                      show=True):
    """
    Plot rollouts in 3D (x,y,z) and (x,y,ω) with a transparent cylindrical obstacle.
    Assumes state = [x, y, z, ω, ...] from simulate_nn().

    Args:
        policy: callable policy used by simulate_nn
        n_steps: steps per rollout
        ngrids: grid size per state dim for initial conditions
        omega_limits: (min,max) for ω axis
        z_limits: (min,max) for z axis
        xy_limits: (min,max) for x and y axes
        out_path: filename stem (no extension)
        save_formats: tuple of extensions to save (e.g., ("svg","pdf","png"))
        show: whether to call plt.show() if a GUI backend is available
    """

    # ---- initial condition grid (x,y,z,ω) around origin; tweak as needed ----
    x0s = []
    grid_axes = [np.linspace(low, high, ngrids) for (low, high) in state_space[:4]]  # first 4 dims: x, y, z, ω
    for i1 in grid_axes[0]:
        for i2 in grid_axes[1]:
            for i3 in grid_axes[2]:
                for i4 in grid_axes[3]:
                    x0s.append([i1, i2, i3, i4])

    # ---- figure ----
    fig = plt.figure(figsize=(12, 6))
    ax_xyz = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xyw = fig.add_subplot(1, 2, 2, projection="3d")

    # ---- simulate and plot trajectories ----
    for x0 in x0s:
        _, x, u = simulate_nn(
            policy,
            x0,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
        )
        x = np.asarray(x)
        if x.shape[1] < 4:
            raise ValueError("Expected state with at least 4 dims: [x, y, z, ω].")

        # (x,y,z)
        ax_xyz.plot(x[:, 0], x[:, 1], x[:, 2], "r-", linewidth=1, alpha=0.9)
        ax_xyz.scatter(x0[0], x0[1], x0[2], color="r", s=6, alpha=0.8)

        # (x,y,ω)
        omega = x[:, 3]
        ax_xyw.plot(x[:, 0], x[:, 1], omega, "b-", linewidth=1, alpha=0.9)
        ax_xyw.scatter(x0[0], x0[1], x0[3], color="b", s=6, alpha=0.8)

    # ---- cylindrical obstacle in xy, extruded along z and ω ----
    # center and radius from your globals
    cx, cy = center[0], center[1]
    r = radius
    r_m = radius + margin

    # helper to draw a transparent cylinder surface
    def draw_cylinder(ax, zmin, zmax, color_edge="k", face_alpha=0.15, edge_alpha=0.4, label=None):
        theta = np.linspace(0, 2*np.pi, 100)
        Z = np.linspace(zmin, zmax, 2)
        TH, ZZ = np.meshgrid(theta, Z)
        X = cx + r * np.cos(TH)
        Y = cy + r * np.sin(TH)
        ax.plot_surface(X, Y, ZZ, linewidth=0, antialiased=False, alpha=face_alpha)
        ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), zmin*np.ones_like(theta),
                color=color_edge, alpha=edge_alpha)
        ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), zmax*np.ones_like(theta),
                color=color_edge, alpha=edge_alpha)
        if label:
            # draw a single ring as legend proxy
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta),
                    ((zmin+zmax)/2)*np.ones_like(theta), color=color_edge, alpha=edge_alpha, label=label)

    def draw_margin(ax, zmin, zmax, style=":", edge_alpha=0.6, label=None):
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(cx + r_m*np.cos(theta), cy + r_m*np.sin(theta), zmin*np.ones_like(theta),
                "k"+style, alpha=edge_alpha, linewidth=1.0)
        ax.plot(cx + r_m*np.cos(theta), cy + r_m*np.sin(theta), zmax*np.ones_like(theta),
                "k"+style, alpha=edge_alpha, linewidth=1.0)
        if label:
            ax.plot(cx + r_m*np.cos(theta), cy + r_m*np.sin(theta),
                    ((zmin+zmax)/2)*np.ones_like(theta), "k"+style, alpha=edge_alpha, linewidth=1.0, label=label)

    # xyz cylinder (extrude along z)
    draw_cylinder(ax_xyz, z_limits[0], z_limits[1], label="Obstacle (radius)")
    draw_margin(ax_xyz, z_limits[0], z_limits[1], label="Safety margin")

    # xyω cylinder (extrude along ω)
    draw_cylinder(ax_xyw, omega_limits[0], omega_limits[1], label="Obstacle (radius)")
    draw_margin(ax_xyw, omega_limits[0], omega_limits[1], label="Safety margin")

    # ---- styling ----
    ax_xyz.set_xlabel("x"); ax_xyz.set_ylabel("y"); ax_xyz.set_zlabel("z")
    ax_xyw.set_xlabel("x"); ax_xyw.set_ylabel("y"); ax_xyw.set_zlabel("ω")

    ax_xyz.set_xlim(xy_limits); ax_xyz.set_ylim(xy_limits); ax_xyz.set_zlim(z_limits)
    ax_xyw.set_xlim(xy_limits); ax_xyw.set_ylim(xy_limits); ax_xyw.set_zlim(omega_limits)

    ax_xyz.set_title("Trajectories in (x, y, z)")
    ax_xyw.set_title("Trajectories in (x, y, ω)")

    # one legend each (uses proxy ring drawn above)
    ax_xyz.legend(loc="upper right")
    ax_xyw.legend(loc="upper right")

    goal_r = 0.25  # radius of goal region
    gx, gy, gz, gw = 1,1,0,0
    theta_c = np.linspace(0, 2*np.pi, 100)
    Xc = gx + goal_r*np.cos(theta_c)
    Yc = gy + goal_r*np.sin(theta_c)
    Zmin, Zmax = ax_xyz.get_zlim()
    Wmin, Wmax = ax_xyw.get_zlim()

    # cylinders in both plots
    for ax, zmin, zmax, label in [
        (ax_xyz, Zmin, Zmax, "Goal region"),
        (ax_xyw, Wmin, Wmax, "Goal region"),
    ]:
        ax.plot(Xc, Yc, zmin*np.ones_like(theta_c), "g--", alpha=0.8)
        ax.plot(Xc, Yc, zmax*np.ones_like(theta_c), "g--", alpha=0.8)
        ax.scatter([gx],[gy],[0 if ax is ax_xyz else gw],color="g",s=30,label=label)


    plt.tight_layout()

    # ---- save outputs ----
    for ext in save_formats:
        fname = f"{out_path}.{ext}"
        try:
            plt.savefig(fname, dpi=300 if ext.lower()=="png" else None)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Could not save {fname}: {e}")

    # ---- show (if GUI backend) or just return ----
    if show:
        plt.show()
    else:
        plt.close(fig)

def save_to_onnx(policy):
    """Save to an onnx file"""
    save_path = "mpc/tests/data/cloned_uni_policy_weight_decay.onnx"
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
