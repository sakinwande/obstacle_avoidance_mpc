"""Define dynamics for systems under study"""
from typing import Callable, List, Union

import casadi
from casadi import cos as ccos, sin as csin, tan as ctan
import numpy as np


# Define a function type for dynamics
DynamicsFunction = Callable[[casadi.MX, casadi.MX], List[casadi.MX]]

# We want our dynamics to be compatible with both casadi and numpy inputs
Variable = Union[casadi.MX, np.ndarray]


def add_dynamics_constraints(
    opti: casadi.Opti,
    dynamics: DynamicsFunction,
    x_now: casadi.MX,
    u_now: casadi.MX,
    x_next: casadi.MX,
    dt: float,
):
    """Add constraints for the dynamics to the given optimization problem

    args:
        opti: the casadi optimization problem to add the constraints to
        dynamics: the function specifying the dynamics (takes current state and control
            input and returns the state derivatives).
        x_now: current state
        u_now: current control input
        x_next: next state
        dt: timestep for Euler integration
    """
    # Get the state derivative
    dx_dt = dynamics(x_now, u_now)

    # Add a constraint for each state variable
    for i in range(x_now.shape[1]):
        xi_next, xi_now, dxi_dt = x_next[i], x_now[i], dx_dt[i]
        opti.subject_to(xi_next == xi_now + dt * dxi_dt)


def dubins_car_dynamics(x: Variable, u: Variable) -> List[Variable]:
    """
    Dubins car dynamics, implemented using Casadi variables

    args:
        x: state variables
        u: control inputs
    """
    # unpack variables
    theta = x[2]
    v = u[0]
    omega = u[1]

    # compute derivatives
    xdot = [
        v * casadi.cos(theta),
        v * casadi.sin(theta),
        omega,
    ]

    return xdot


def quad6d_dynamics(x: Variable, u: Variable):
    """
    Nonlinear 6D quadrotor dynamics, implemented using Casadi variables

    args:
        x: state variables
        u: control inputs [theta, phi, tau]. Tau is defined as the excess/surplus
            acceleration relative to gravity.
    """
    # unpack variables
    vx = x[3]
    vy = x[4]
    vz = x[5]
    theta = u[0]
    phi = u[1]
    tau = u[2]

    # compute derivatives
    g = 9.81
    xdot = [
        vx,
        vy,
        vz,
        g * casadi.tan(theta),
        -g * casadi.tan(phi),
        tau,
    ]

    return xdot

def quad_6d_dynamics_nonlinear(x: Variable, u: Variable):
    """
    Nonlinear 6D quadrotor dynamics, implemented inspired by archcomp

    args:
        x: state variables
        u: control inputs are basically velocities (linear and angular) 
           
    """
    # unpack variables
    x1 = x[0] #inertial (north) position
    x2 = x[1] #inertial (east) position
    x3 = x[2] ##altitude
    x4 = x[3] #roll angle
    x5 = x[4] #pitch angle
    x6 = x[5] #yaw angle
    u1 = u[0] #x position control (treated as velocity)
    u2 = u[1] #y position control (treated as velocity)
    u3 = u[2] #z position control (treated as velocity)
    u4 = u[3] #roll angle control (treated as velocity)
    u5 = u[4] #pitch angle control (treated as velocity)
    u6 = u[5] #yaw angle control (treated as velocity)

    # compute derivatives
    xdot = [
        u1 + csin(x4)*csin(x5)*ccos(x6) - ccos(x4)*csin(x6) + ccos(x4)*csin(x5)*ccos(x6) + csin(x4)*csin(x6),
        u2 + ccos(x5)*csin(x6) + ccos(x4)*csin(x5)*csin(x6) - csin(x4)*ccos(x6),
        -u3 + csin(x5) - csin(x4)*ccos(x5),
        u4 + csin(x4)*ctan(x5) + ccos(x4)*ctan(x5),
        u5 - csin(x4),
        csin(x4)/ccos(x5) - u6
    ]
    return xdot

def attitude_control_dynamics(x: Variable, u: Variable):
    """
    Nonlinear rigid body dynamics for attitude control.

    Basically a 6D nonlinear model

    args:
        x: state variables
        u: control inputs
    """

    # unpack variables
    phi_1 = x[0] #rodrigues parameter 1, akin to rotation about x axis
    phi_2 = x[1] #rodrigues parameter 2, akin to rotation about y axis
    phi_3 = x[2] #rodrigues parameter 3, akin to rotation about z axis 
    omega_1 = x[3] #body frame angular velocity about x axis
    omega_2 = x[4] #body frame angular velocity about y axis
    omega_3 = x[5] #body frame angular velocity about z axis

    dPhi_1 = 0.5*(omega_2*(phi_1**2 + phi_2**2 + phi_3**2 - phi_3) + omega_3*(phi_1**2 + phi_2**2 + phi_2 + phi_3**2) + omega_1 * (phi_1**2 + phi_2**2 + phi_3**2 +1))

    dPhi_2 = 0.5*(omega_1*(phi_1**2 + phi_2**2 + phi_3**2 + phi_3) + omega_3 * (phi_1**2 - phi_1 + phi_2**2 + phi_3**2) + omega_2 * (phi_1**2 + phi_2**2 + phi_3**2 +1))

    dPhi_3 = 0.5*(omega_1*(phi_1**2 + phi_2**2 - phi_2 + phi_3**2) + omega_2*(phi_1**2 + phi_1 + phi_2**2  + phi_3**2) + omega_3 * (phi_1**2 + phi_2**2 + phi_3**2 +1))

    dOmgega_1 = 0.25 * (u[0] + omega_2*omega_3)
    dOmega_2 = 0.5 * (u[1] - 3*omega_3*omega_1)
    dOmega_3 = u[2] + 2*omega_1*omega_2

    xdot = [dPhi_1, dPhi_2, dPhi_3, dOmgega_1, dOmega_2, dOmega_3]

    return xdot

def unicycle_dynamics(x: Variable, u: Variable):
    # unpack variables
    x1 = x[0] #Cartesian x position
    x2 = x[1] #Cartesian y position
    x3 = x[2] #Steering angle
    x4 = x[3] #Forward velocity magnitude
    u1 = u[0] #Velocity command
    u2 = u[1] #Stering angle command 

    xdot = [
        x4*ccos(x3),
        x4*csin(x3),
        u2,
        u1 
    ]
    return xdot

def tora_dynamics(x: Variable, u: Variable):
    # unpack variables
    x1 = x[0] 
    x2 = x[1] 
    x3 = x[2]
    x4 = x[3] 
    u1 = u[0] 

    xdot = [
        x2,
        -x1 + 0.1*csin(x3),
        x4,
        u1 
    ]
    return xdot


def quad12d_dynamics(x: Variable, u: Variable):
    """
    Nonlinear 12D quadrotor dynamics, implemented using Casadi variables

    args:
        x: state variables
        u: control inputs [theta, phi, tau]. Tau is defined as the excess/surplus
            acceleration relative to gravity.
    """
    # unpack variables
    x1 = x[0] #inertial (north) position
    x2 = x[1] #inertial (east) position
    x3 = x[2] ##altitude
    x4 = x[3] #longitudinal velocity
    x5 = x[4] #lateral velocity
    x6 = x[5] #vertical velocity
    x7 = x[6] #roll angle
    x8 = x[7] #pitch angle
    x9 = x[8] #yaw angle
    x10 = x[9] #roll rate
    x11 = x[10] #pitch rate
    x12 = x[11] #yaw rate
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]

    #define constants
    g = 9.81
    m = 1.4
    Jx = 0.054
    Jy = 0.054
    Jz = 0.104
    tauPhi = 0.0

    # compute derivatives
    xdot = [
        ccos(x8)*ccos(x9)*x4 + (csin(x7)*csin(x8)*ccos(x9) - ccos(x7)*csin(x9))*x5 + (ccos(x7)*csin(x8)*ccos(x9) + csin(x7)*csin(x9))*x6,
        ccos(x8)*csin(x9)*x4 + (csin(x7)*csin(x8)*csin(x9) + ccos(x7)*ccos(x9))*x5 + (ccos(x7)*csin(x8)*csin(x9) - csin(x7)*ccos(x9))*x6,
        csin(x8)*x4 - csin(x7)*ccos(x8)*x5 - ccos(x7)*ccos(x8)*x6,
        x12*x5 - x11*x6 - g*csin(x8),
        x10*x6 - x12*x4 + g*csin(x7)*ccos(x8),
        x11*x4 - x10*x5 + g*ccos(x7)*ccos(x8) -g -u1/m,
        x10 + csin(x7)*ctan(x8)*x11 + ccos(x7)*ctan(x8)*x12,
        ccos(x7)*x11 - csin(x7)*x12,
        (csin(x7)/ccos(x8))*x11 + (ccos(x7)/ccos(x8))*x12,
        ((Jy-Jz)/Jx)*x11*x12 + u2/Jx,
        ((Jz-Jx)/Jy)*x10*x12 + u3/Jy,
        ((Jx-Jy)/Jz)*x10*x11 + tauPhi/Jz,
    ]

    return xdot