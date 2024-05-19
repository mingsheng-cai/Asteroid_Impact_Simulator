"""Contains the atmospheric entry solver class for the Deep Impact project."""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .ode_solver import OdeSolver

__all__ = ["Planet"]


class Planet:
    """Solve the atmospheric entry problem for a given planet.

    The class called Planet is initialised with constants appropriate for
    the given target planet, including the atmospheric density profile and
    other constants.
    """

    def __init__(
        self,
        atmos_func="exponential",
        atmos_filename=os.sep.join(
            (
                os.path.dirname(__file__),
                "..",
                "resources",
                "AltitudeDensityTable.csv",
            )
        ),
        Cd=1.0,
        Ch=0.1,
        Q=1e7,
        Cl=1e-3,
        alpha=0.3,
        Rp=6371e3,
        g=9.81,
        H=8000.0,
        rho0=1.2,
    ):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """
        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == "exponential":
                self.rhoa = lambda z: rho0 * np.exp(-z / H)
            elif atmos_func == "tabular":
                densityValues = np.genfromtxt(self.atmos_filename)
                altitude, density = densityValues[:, 0], densityValues[:, 1]

                _interp1d = interp1d(
                    altitude,  # altitude in m
                    density,  # density in kg/m^3
                    kind="cubic",
                    bounds_error=False,
                    fill_value=(density[0], density[-1]),
                )

                self.rhoa = lambda y: _interp1d(y).item()
            elif atmos_func == "constant":
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def change_time_step(
        self, df_raw, time_col, original_time_step, new_time_step
    ):
        """
        Adjust the time step granularity of a DataFrame.

        This method changes the granularity of the time
        steps in a DataFrame, either by upsampling or
        downsampling, based on the specified original
        and new time step values. It handles both
        increasing and decreasing of time step granularity
        through resampling or interpolating methods, respectively.

        Parameters:
        df_raw (pd.DataFrame): original DataFrame with time series data.
        time_col (str): column name in df_raw that contains the time steps.
        original_time_step (float): time step (seconds) of original data.
        new_time_step (float): desired time step (seconds) for new data.

        Returns:
        pd.DataFrame: new DataFrame with the adjusted time step granularity.
        The original time step column is converted to a timedelta index,
        and the transformation is applied based on the new time step value
        compared to the original. The method returns the original DataFrame
        if the new time step is equal to the original time step.

        Notes:
        - Upsampling (new_time_step > original_time_step)
          involves resampling with mean aggregation.
        - Downsampling (new_time_step < original_time_step)
          involves reindexing with linear interpolation.
        """
        # Determine the Pandas frequency string
        # for the original and new time steps
        new_freq = f"{new_time_step}S"
        df = df_raw.copy()

        # Convert the time step column to a timedelta index
        df[time_col] = pd.to_timedelta(df[time_col], unit="s")
        df.set_index(time_col, inplace=True)

        if float(new_time_step) == float(original_time_step):
            return df_raw
        elif new_time_step > original_time_step:
            # Increase time step granularity
            df_resampled = df.resample(new_freq).mean()
            df_resampled.reset_index(inplace=True)
            df_resampled["time"] = df_resampled["time"].dt.total_seconds()
            return df_resampled
        elif new_time_step < original_time_step:
            # Decrease time step granularity
            new_time_range = pd.timedelta_range(
                start=df.index.min(), end=df.index.max(), freq=new_freq
            )
            df_reindexed = df.reindex(new_time_range)
            df_interpolated = df_reindexed.interpolate(method="linear")
            df_interpolated.reset_index(inplace=True)
            df_interpolated.rename(columns={"index": time_col}, inplace=True)
            df_interpolated["time"] = df_interpolated[
                "time"
            ].dt.total_seconds()
            return df_interpolated

    def dy_dt(self, y, fractioned_phase, density):
        """
        Return the derivative of y at time t.

        Parameters
        ----------
        y : array_like
            An array containing the current values of the variables
            [x, vel, z, theta, mass, radius]
        fractioned_phase : int
            Phase of the trajectory. This can be:
                - 0: non fractioned
                - 1: fractioned
                - 2: disappeared / reached ground
        density : float
            The density of the asteroid in kg/m^3
        """
        _, vel, z, theta, mass, radius = y

        if fractioned_phase == 0:
            radius = (3.0 * mass / (4.0 * np.pi * density)) ** (1.0 / 3.0)

        A = np.pi * radius**2

        dm_dt = 0.0
        dr_dt = 0.0
        dtheta_dt = 0.0
        dx_dt = vel * np.cos(theta)
        dz_dt = -vel * np.sin(theta)
        dv_dt = -self.Cd * self.rhoa(z) * (vel**2) * A / (2 * mass)

        if not (
            np.isclose(0.0, 2 * self.Q)
            and np.isclose(0.0, self.g)
            and np.isclose(0.0, self.Cl)
        ):
            dx_dt /= 1 + z / self.Rp
            dv_dt += self.g * np.sin(theta)
            dm_dt = -self.Ch * self.rhoa(z) * (vel**3) * A / (2 * self.Q)
            dtheta_dt = (
                self.g * np.cos(theta) / vel
                - self.Cl * self.rhoa(z) * vel * A / (2 * mass)
                - vel * np.cos(theta) / (self.Rp + z)
            )

            if fractioned_phase == 1:
                dr_dt = (
                    np.sqrt(7.0 / 2.0 * self.alpha * self.rhoa(z) / density)
                    * vel
                )

        return np.array([dx_dt, dv_dt, dz_dt, dtheta_dt, dm_dt, dr_dt])

    def solve_atmospheric_entry(
        self,
        radius,
        velocity,
        density,
        strength,
        angle,
        init_altitude=100e3,
        dt=0.05,
        dt_ode=0.05,
        radians=False,
        ode_method="rk4",
    ):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entry speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        dt_ode : float, optional
            The timestep for the ODE solver, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        ode_method : string, optional
            The method to use for solving the ODE. Default is 'improved_euler'.
            Options are 'forward_euler', 'improved_euler', 'rk4'

        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """
        dt_ode = min(dt, dt_ode)  # select the time step for ode solver
        theta0 = angle if radians else np.deg2rad(angle)

        t0 = 0.0
        x0 = 0.0
        vel0 = velocity
        radius0 = radius
        z0 = init_altitude
        mass0 = (4.0 / 3.0) * np.pi * density * radius**3
        y0 = np.array([x0, vel0, z0, theta0, mass0, radius0])

        def has_fractioned_func(w):
            test_frac = self.rho0 * np.exp(-w[2] / self.H) * w[1] ** 2
            return test_frac > strength

        results_y, results_time = OdeSolver(ode_method)(
            lambda _, y, fractioned_phase: self.dy_dt(
                y, fractioned_phase, density
            ),
            y0,
            t0,
            dt_ode,
            lambda y: y[2] <= 0.0,  # reached_ground
            has_fractioned_func,  # fractioned
            lambda y: self.rho0 * np.exp(-y[2] / self.H) * y[1] ** 2
            < strength,  # has disappeared
        )

        raw_data = {
            "velocity": results_y[:, 1],
            "mass": results_y[:, 4],
            "altitude": results_y[:, 2],
            "distance": results_y[:, 0],
            "radius": results_y[:, 5],
            "time": results_time[:],
            "angle": results_y[:, 3]
            if radians
            else np.rad2deg(results_y[:, 3]),
        }

        return self.change_time_step(
            pd.DataFrame(raw_data), "time", dt_ode, dt
        )

    def calculate_energy(self, result):
        """Calculate the kinetic energy as a function of time.

        Function to calculate the kinetic energy lost per unit
        altitude in kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """
        result["Ek"] = 0.5 * result.mass * result.velocity**2

        if len(result) == 1:
            result["dedz"] = 0
            return result
        elif len(result) == 0:
            return result

        dedz_approx_list = [0]
        for i in range(1, len(result) - 1):
            # takes the central difference approximation
            # of the derivative in the kinetic energy
            delta_i = result.loc[i + 1, "Ek"] - result.loc[i - 1, "Ek"]

            # takes the central difference approximation
            # of the derivative in the altitude
            delta_j = (
                result.loc[i + 1, "altitude"] - result.loc[i - 1, "altitude"]
            )

            dedz_approx_list.append(delta_i / delta_j)

        # derivative of the last point calculated using the backward difference
        ek_current_val = result.loc[len(result) - 1, "Ek"]
        ek_forward_val = result.loc[(len(result) - 2) % len(result), "Ek"]

        altitude_current_val = result.loc[len(result) - 1, "altitude"]
        altitude_forward_val = result.loc[
            (len(result) - 2) % len(result), "altitude"
        ]

        ek_end_derivative = (ek_current_val - ek_forward_val) / (
            altitude_current_val - altitude_forward_val
        )

        dedz_approx_list.append(ek_end_derivative)

        result["dedz"] = dedz_approx_list
        result["dedz"] = result["dedz"].apply(
            lambda x: 1000 * x / (4.184 * 10**12)
        )
        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            A pandas DataFrame containing data about an
            event over time, including velocity, mass,
            angle, altitude, horizontal distance, radius, and dedz.

        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key:
                ``outcome`` (which should contain one of the
                following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
                ``burst_peak_dedz``, ``burst_altitude``,
                ``burst_distance``, ``burst_energy``
        """
        idx_max_dedz = result.dedz.idxmax()
        if result.loc[idx_max_dedz, "altitude"] < 0.0:
            if idx_max_dedz == 0:
                return self._compute_output_dict(result, 0, "Cratering")
            return self._analyze_cratering(result, idx_max_dedz)
        return self._compute_output_dict(result, idx_max_dedz, "Airburst")

    def _compute_output_dict(self, result, idx: int, output: str) -> dict:
        """Compute the output dictionary for the given event.

        Parameters
        ----------
        result : DataFrame
            The result DataFrame containing data of the event.
        idx : int
            Index in the DataFrame where the event occurred.

        Returns
        -------
        dict
            A dictionary containing the event metrics.
        """
        E0 = 0.5 * result.loc[0, "mass"] * result.loc[0, "velocity"] ** 2
        Eh = 0.5 * result.loc[idx, "mass"] * result.loc[idx, "velocity"] ** 2

        return {
            "outcome": output,
            "burst_peak_dedz": result.loc[idx, "dedz"],
            "burst_altitude": result.loc[idx, "altitude"],
            "burst_distance": result.loc[idx, "distance"],
            "burst_energy": max((E0 - Eh), Eh) / (4.184 * 1e12),
        }

    def _analyze_airburst(self, result, idx: int) -> dict:
        """
        Analyze the airburst event.
        """
        return self._compute_output_dict(result, idx, "Airburst")

    def _analyze_cratering(self, result, idx: int) -> dict:
        """
        Calculate the cratering event metrics using linear interpolation.

        Parameters
        ----------
        result : DataFrame
            The result DataFrame containing data of the event.
        idx : int
            Index in the DataFrame where cratering metrics are calculated.

        Returns
        -------
        dict
            A dictionary containing interpolated values of event metrics
            at the point of cratering, including burst energy.
        """
        # Extract relevant data for current and previous time steps
        alt_curr = result.loc[idx, "altitude"]
        alt_prev = result.loc[idx - 1, "altitude"]
        time_prev, time_curr = (
            result.loc[idx - 1, "time"],
            result.loc[idx, "time"],
        )
        metrics_curr = result.loc[
            idx, ["distance", "dedz", "altitude", "velocity", "mass"]
        ]
        metrics_prev = result.loc[
            idx - 1, ["distance", "dedz", "altitude", "velocity", "mass"]
        ]

        # Calculate zero altitude time using linear interpolation
        zero_altitude_time = (alt_curr * time_prev - alt_prev * time_curr) / (
            alt_curr - alt_prev
        )

        # Interpolate other values at the zero altitude time
        def interpolate_func(key):
            # Calculate the slope (rate of change)
            slope = (metrics_curr[key] - metrics_prev[key]) / (
                time_curr - time_prev
            )

            # Use the slope for linear interpolation
            # to estimate the value at zero_altitude_time
            return slope * (zero_altitude_time - time_prev) + metrics_prev[key]

        # Calculate burst energy
        df = pd.DataFrame(
            {key: interpolate_func(key) for key in metrics_curr.keys()},
            index=[0],
        )

        return self._compute_output_dict(df, 0, "Cratering")
