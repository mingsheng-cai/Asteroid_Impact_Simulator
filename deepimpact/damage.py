"""Module to calculate the damage and impact risk for given scenarios."""
import pandas as pd
import numpy as np
import sympy as sp
import os

from deepimpact.locator import GeospatialLocator

__all__ = ["damage_zones", "impact_risk"]


def damage_zones(outcome, lat, lon, bearing, pressures):
    """Calulate latitude and longitude of surface zero and blast radii.

    Calculate the latitude and longitude of the surface zero location and
    the list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------
    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------
    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------
    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, \
    'burst_energy': 7e3, 'burst_distance': 90e3, \
    'burst_peak_dedz': 1e3, 'outcome': 'Airburst'}
    >>> deepimpact.damage_zones(outcome, 52.79, -2.95, 135, \
    pressures=[1e3, 3.5e3, 27e3, 43e3])
    (52.79, -2.95, [5000.0, 5000.0, 5000.0, 5000.0])
    """

    burst_distance = outcome["burst_distance"]
    burst_energy = outcome["burst_energy"]
    burst_altitude = outcome["burst_altitude"]

    blat, blon = calculate_surface_zero(lat, lon, bearing, burst_distance)

    damrad = calculate_damage_radius(pressures, burst_energy, burst_altitude)

    return blat, blon, damrad


def impact_risk(
    planet,
    impact_file=os.sep.join(
        (
            os.path.dirname(__file__),
            "..",
            "resources",
            "impact_parameter_list.csv",
        )
    ),
    pressure=30.0e3,
    nsamples=None,
):
    """Calculate the probability of impact for each UK postcode.

    Perform an uncertainty analysis to calculate the probability for each
    affected UK postcode and the total population affected.

    Parameters
    ----------
    planet: deepimpact.Planet instance
        The Planet instance from which to solve the atmospheric entry

    impact_file: str
        Filename of a .csv file containing the impact parameter list
        with columns for 'radius', 'angle', 'velocity', 'strength',
        'density', 'entry latitude', 'entry longitude', 'bearing'

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int or None
        The number of iterations to perform in the uncertainty analysis.
        If None, the full set of impact parameters provided in impact_file
        is used.

    Returns
    -------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.
    """

    # Read the impact parameter list from the file
    impact_parameters = pd.read_csv(impact_file)

    # If nsamples is not None, sample the impact parameters
    if nsamples is not None and nsamples < len(impact_parameters):
        impact_parameters = impact_parameters.sample(n=nsamples)

    # Initialise the GeospatialLocator
    locator = GeospatialLocator()

    # Initialise the dictionaries to store the postcode counts and population
    postcode_counts = {}
    population_list = []

    # Iterate over the impact parameters
    for index, row in impact_parameters.iterrows():
        # Solve the atmospheric entry problem for the current impact parameters
        result = planet.solve_atmospheric_entry(
            radius=row["radius"],
            angle=row["angle"],
            strength=row["strength"],
            density=row["density"],
            velocity=row["velocity"],
        )
        result = planet.calculate_energy(result)
        outcome = planet.analyse_outcome(result)

        # Calculate the damage radius for the current pressure
        blat, blon, damrad = damage_zones(
            outcome,
            row["entry latitude"],
            row["entry longitude"],
            row["bearing"],
            [pressure],
        )

        # Find the postcodes in the damage radius
        affected_postcodes = locator.get_postcodes_by_radius(
            (blat, blon), damrad
        )

        # Count the number of times each postcode appears
        for postcode_list in affected_postcodes:
            for postcode in postcode_list:
                if postcode in postcode_counts:
                    postcode_counts[postcode] += 1
                else:
                    postcode_counts[postcode] = 1

        # Find the population in the damage radius
        population_in_this_scenario = locator.get_population_by_radius(
            (blat, blon), damrad
        )
        population_list.extend(population_in_this_scenario)

    # Calculate the probability of each postcode being affected
    total_scenarios = len(impact_parameters)
    probability_data = [
        {"postcode": postcode, "probability": count / total_scenarios}
        for postcode, count in postcode_counts.items()
    ]

    # Create a pandas DataFrame from the probability data
    probability = pd.DataFrame(probability_data)

    # Calculate the mean and standard deviation of the population affected
    population_mean = float(np.mean(population_list))
    population_stdev = float(np.std(population_list))

    population = {"mean": population_mean, "stdev": population_stdev}

    return probability, population


def calculate_surface_zero(lat, lon, bearing, distance):
    """
    Calculate the latitude and longitude of the surface zero location for a.

    Parameters
    ----------

    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    distance : float
        Distance from the meteoroid entry point to the surface zero point (m)

    Returns
    -------

    blat : _type_
        latitude of the surface zero point
    blon : _type_
        longitude of the surface zero point
    """

    R = 6371000.0

    # Convert input parameters to radians
    _lat, _, _bearing = (
        np.radians(lat),
        np.radians(lon),
        np.radians(bearing),
    )

    _blat = np.arcsin(
        np.sin(_lat) * np.cos(distance / R)
        + np.cos(_lat) * np.sin(distance / R) * np.cos(_bearing)
    )
    _blon_minus_lon = np.arctan(
        np.sin(_bearing)
        * np.sin(distance / R)
        * np.cos(_lat)
        / (np.cos(distance / R) - np.sin(_lat) * np.sin(_blat))
    )

    blat = float(np.degrees(_blat))
    blon = float(np.degrees(_blon_minus_lon) + lon)

    return blat, blon


def calculate_damage_radius(pressures, energy, altitude):
    """calculate the damage radius for a given set of pressures

    Parameters
    ----------

    pressures : list
        List of pressures to calculate the damage radius for
    energy : float
        Energy of the meteoroid
    altitude : float
        Altitude of the meteoroid

    Returns
    -------

    radii: float, list
        List of damage radii for the given pressures
    """

    radii = []

    e_val = energy
    z_val = altitude

    is_pressures_float = isinstance(pressures, float)
    pressures = np.atleast_1d(pressures)

    initial_interval = (1e-10, 1e10)

    for p_val in pressures:
        equation_to_solve = (
            lambda r: 3
            * 10**11
            * ((r**2 + z_val**2) / e_val ** (2 / 3)) ** (-1.3)
            + 2
            * 10**7
            * ((r**2 + z_val**2) / e_val ** (2 / 3)) ** (-0.57)
            - p_val
        )

        initial_guess = bisect_for_initial_guess(
            equation_to_solve, initial_interval
        )

        radius = approach_radius_by_newton_raphson(
            p_val, e_val, z_val, initial_guess
        )

        radii.append(radius)

    if is_pressures_float:
        radii = radii[0]

    return radii


def bisect_for_initial_guess(func, interval, tolerance=1e-8, max_iter=100):
    """
    Find the initial guess for the radius of the meteoroid

    Parameters
    ----------

    func : function
        Function to find the initial guess for
    interval : turple
        Interval to find the initial guess in
    tolerance : float, optional
        Tolernace of the initial guess, by default 1e-8
    max_iter : int, optional
        Maximum number of iterations, by default 100

    Returns
    -------

    float
        Initial guess for the radius of the meteoroid
    """

    a, b = interval

    for _ in range(max_iter):
        mid_point = (a + b) / 2
        if func(mid_point) == 0 or (b - a) / 2 < tolerance:
            return mid_point
        elif np.sign(func(mid_point)) == np.sign(func(a)):
            a = mid_point
        else:
            b = mid_point

    return (a + b) / 2


def approach_radius_by_newton_raphson(
    p_val, e_val, z_val, r_guess, tolerance=1e-8, max_iter=100
):
    """
    Approach the radius of the meteoroid using Newton-Raphson method

    Parameters
    ----------

    p_val : float
        Pressure value
    e_val : float
        Energy value
    z_val : float
        Altitude value
    r_guess : float
        Initial guess for the radius of the meteoroid
    tolerance : float, optional
        Tolernace of the radius, by default 1e-8
    max_iter : int, optional
        Maximum number of iterations, by default 100

    Returns
    -------

    float
        Radius of the meteoroid

    Acknowledgements
    ----------------

        ChatGPT4
        Github Copilot
    """

    r, z, e, p = sp.symbols("r z e p")

    expression = (
        3 * 10**11 * ((r**2 + z**2) / e ** (2 / 3)) ** (-1.3)
        + 2 * 10**7 * ((r**2 + z**2) / e ** (2 / 3)) ** (-0.57)
        - p
    )

    derivative = sp.diff(expression, r)

    next_iteration = (
        lambda r_value, z_value, e_value, p_value: r_value
        - expression.subs({r: r_value, z: z_value, e: e_value, p: p_value})
        / derivative.subs({r: r_value, z: z_value, e: e_value, p: p_value})
    )

    for i in range(max_iter):
        r_next = next_iteration(r_guess, z_val, e_val, p_val)
        if abs(r_next - r_guess) < tolerance:
            break
        r_guess = r_next

    return float(r_next)
