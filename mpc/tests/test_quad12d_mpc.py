"""Test the obstacle avoidance MPC for a quadrotor"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
    squared_error_terminal_cost,
    zero_running_cost,
)
from mpc.dynamics_constraints import quad12d_dynamics
from mpc.mpc import construct_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_mpc


radius = 0.5
margin = 0.1
#Define the center of the sphere to be far from op region
center = [1.8 for _ in range(12)]
center[0] = 0.0
#Define the indices of the state that are in the sphere
indices = [i for i in range(12)]



def test_quad_mpc(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a test of obstacle avoidance MPC with a quad and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    n_states = 12
    n_controls = 3
    horizon = 50
    dt = 0.1

    # Define dynamics
    dynamics_fn = quad12d_dynamics

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, indices, center), margin)]

    # Define costs to make the quad up
    x_goal = np.zeros(12)
    x_goal[2] = 1.000
    #We're just going up
    goal_direction = np.zeros(12)
    goal_direction[2] = 1.0
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([0.1, 0.1, 10, 0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1]), 0.1* np.eye(3)
    )
    # running_cost_fn = lambda x, u: zero_running_cost(x, u)
    # terminal_cost_fn = lambda x: distance_travelled_terminal_cost(x, goal_direction)
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal,dt * np.diag([0.1, 0.1, 10.0, 0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1]))

    # Define control bounds
    control_bounds = [np.pi / 10, np.pi / 10, 2.0]

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

    # -------------------------------------------
    # Simulate and return the results
    # -------------------------------------------
    n_steps = 50
    return simulate_mpc(
        opti,
        x0_variables,
        u0_variables,
        x0,
        dt,
        dynamics_fn,
        n_steps,
        verbose=False,
        x_variables=x_variables,
        u_variables=u_variables,
        substeps=10,
    )


def run_and_plot_quad_mpc():
    ngrids = 2
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

    for x0 in x0s:
        # Run the MPC
        _, x, u = test_quad_mpc(x0)

        # Plot it (in x-y plane)
        # ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        # ax_xz.plot(x0[0], x0[2], "ro")
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

    #Plot goal region in x-z plane
     

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-5.0, 5.0])
    ax_xy.set_ylim([-5.0, 5.0])
    ax_xz.set_xlim([-5.0, 5.0])
    ax_xz.set_ylim([-2.0, 2.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.show()
    b = 1


def plot_sdf():
    sdf_fn = lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center)
    xs = np.linspace(-1.0, 1.0, 200)
    ys = np.linspace(-1.0, 1.0, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            state = np.array([[x, y, 0.0, 0.0, 0.0, 0.0]])
            sdf = sdf_fn(state)
            Z[j, i] = np.exp(1e2 * (margin - sdf))

    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    run_and_plot_quad_mpc()
    b=2
    # plot_sdf()
