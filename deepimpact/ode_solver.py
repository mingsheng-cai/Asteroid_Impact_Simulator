"""Module to solve a system of ODEs using the selected method."""
import numpy as np

__all__ = ["OdeSolver"]


class OdeSolver:
    """
    Solve a system of ODEs using the selected method.

    Parameters
    ----------
    method(str): method to use to solve the ODE system. This can be:
        - "improved_euler": improved Euler method
        - "forward_euler": forward Euler method
        - "rk4": Runge-Kutta 4th order method
    """

    def __init__(self, method):
        if method == "improved_euler":
            self.step_solver_t = OdeSolver.improved_euler_step_t
        elif method == "forward_euler":
            self.step_solver_t = OdeSolver.forward_euler_step_t
        else:
            self.step_solver_t = OdeSolver.RK4_step_t

    def __call__(
        self,
        f,
        y0,
        t0,
        dt,
        has_reached_ground,
        has_fractioned,
        has_disappeared,
    ):
        """
        Solve the ODE system using the selected method.

        Parameters
        ----------
        f(function): function to compute the right hand side of the ODE
        y0(array_like): initial state
        t0(float): initial time
        dt(float): time step
        has_reached_ground(function): function to check if the projectile has
            reached the ground

        Returns
        -------
        y_all(array_like): an array of the states at each time (t)
        t_all(array_like): an array of times for which the ODE system
            is solved
        """
        y = np.array(y0)
        t = np.array(t0)
        y_all = [y0]
        t_all = [t0]

        # Conditions to stop the solver:
        # 1) if z becomes <= 0
        # 2) if `self.rho0 * np.exp(-y[2] / self.H) * y[1]**2 < strength`
        fractioned_count = 0
        fractioned_phase = 0

        previous_y = y

        while not has_reached_ground(y):
            if len(y_all) > 1:
                if y_all[-2][2] < y_all[-1][2]:
                    break
            if (
                fractioned_phase == 0
                and has_fractioned(y)
                and fractioned_count < 1
            ):
                fractioned_phase = 1
                fractioned_count += 1
            elif fractioned_phase == 1 and has_disappeared(y):
                fractioned_phase = 0

            y = self.step_solver_t(t, y, f, dt, fractioned_phase)

            # if y has not changed, then it will never change
            # of course, this is valid only because the RHS odes
            # are not time dependent
            if np.allclose(y, previous_y):
                break
            previous_y = y

            t = t + dt
            y_all.append(y)
            t_all.append(t)
        return np.array(y_all), np.array(t_all)

    # TODO: add tests for this method
    def improved_euler_step_t(t, y, f, dt, fractioned_phase):
        """
        Improved Euler step for a single variable.

        Parameters
        ----------
        t(float): current time
        y(array_like): current state
        f(function): function to compute the right hand side of the ODE
        dt(float): time step
        fractioned_phase(int): phase of the trajectory. This can be:
            - 0: non fractioned
            - 1: fractioned
            - 2: disappeared / reached ground

        Returns
        -------
        y(array_like): state at t = t + dt obtained by the improved Euler
            method
        """
        ye = y + dt * f(t, y, fractioned_phase)  # euler guess
        y = y + 0.5 * dt * (
            f(t, y, fractioned_phase) + f(t + dt, ye, fractioned_phase)
        )
        return y

    def forward_euler_step_t(t, y, f, dt, fractioned_phase):
        """
        Forward Euler step for a single variable.

        Parameters
        ----------
        t(float): current time
        y(array_like): current state
        f(function): function to compute the right hand side of the ODE
        dt(float): time step
        fractioned_phase(int): phase of the trajectory. This can be:
            - 0: non fractioned
            - 1: fractioned
            - 2: disappeared / reached ground

        Returns
        -------
        y(array_like): state at t = t + dt obtained by the forward Euler
            method
        """
        y = y + dt * f(t, y, fractioned_phase)  # euler guess
        return y

    def RK4_step_t(t, y, f, dt, fractioned_phase):
        """
        Runge-Kutta 4th order step for a single variable.

        Parameters
        ----------
        t(float): current time
        y(array_like): current state
        f(function): function to compute the right hand side of the ODE
        dt(float): time step
        fractioned_phase(int): phase of the trajectory. This can be:
            - 0: non fractioned
            - 1: fractioned
            - 2: disappeared / reached ground

        Returns
        -------
        y(array_like): state at t = t + dt obtained by the RK4 method
        """
        k1 = dt * f(t, y, fractioned_phase)
        k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1, fractioned_phase)
        k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2, fractioned_phase)
        k4 = dt * f(t + dt, y + k3, fractioned_phase)
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y
