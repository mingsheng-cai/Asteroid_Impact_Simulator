import numpy as np
from scipy.optimize import fsolve

from pytest import fixture


@fixture(scope="module")
def deepimpact():
    import deepimpact

    return deepimpact


@fixture(scope="module")
def planet(deepimpact):
    return deepimpact.Planet()


@fixture(scope="module")
def result(planet):
    input = {
        "radius": 1.0,
        "velocity": 2.0e4,
        "density": 3000.0,
        "strength": 1e5,
        "angle": 30.0,
        "init_altitude": 0.0,
    }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope="module")
def outcome(planet, result):
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_value_type_of_damage_zones(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    pressures = [27e3, 43e3]

    lat, lon, damrad = deepimpact.damage_zones(
        outcome, 54.32, 1.23, 135.0, pressures
    )

    assert isinstance(lat, float)
    assert isinstance(lon, float)

    assert isinstance(damrad, list)

    for i in range(len(damrad)):
        assert isinstance(damrad[i], float)


def test_radii_value_of_damage_zones(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    pressures = [27e3, 43e3]

    true_value = []

    for p in pressures:
        equation_to_solve = (
            lambda r: 3
            * 10**11
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-1.3)
            + 2
            * 10**7
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-0.57)
            - p
        )
        initial_guess = 1.0
        result = fsolve(equation_to_solve, initial_guess)
        true_value.append(result[0])

    lat, lon, damrad = deepimpact.damage_zones(
        outcome, 55.0, 0.0, 135.0, pressures
    )

    assert np.isclose(lat, 54.42365982774329)
    assert np.isclose(lon, 0.983751262482494)

    assert len(true_value) == len(damrad)

    for i in range(len(true_value)):
        assert np.isclose(damrad[i], true_value[i])


def test_radii_value_of_damage_zones_(deepimpact):
    outcome = {
        "burst_altitude": 8e3,
        "burst_energy": 7e3,
        "burst_distance": 90e3,
        "burst_peak_dedz": 1e3,
        "outcome": "Airburst",
    }

    pressures = [1e3, 3.5e3, 27e3, 43e3]

    true_value = []

    for p in pressures:
        equation_to_solve = (
            lambda r: 3
            * 10**11
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-1.3)
            + 2
            * 10**7
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-0.57)
            - p
        )
        initial_guess = 1.0
        result = fsolve(equation_to_solve, initial_guess)
        true_value.append(result[0])

    lat, lon, damrad = deepimpact.damage_zones(
        outcome, 52.79, -2.95, 135, pressures
    )

    assert np.isclose(lat, 52.21396905216966)
    assert np.isclose(lon, -2.015908861677074)

    assert len(true_value) == len(damrad)

    for i in range(len(true_value)):
        assert np.isclose(damrad[i], true_value[i])


def test_radii_value_of_damage_zones__(deepimpact):
    outcome = {
        "burst_altitude": 8e3,
        "burst_energy": 7e3,
        "burst_distance": 90e3,
        "burst_peak_dedz": 1e3,
        "outcome": "Airburst",
    }

    pressures = [
        1e3,
        2e3,
        3e3,
        4e3,
        5e3,
        6e3,
        7e3,
        8e3,
        9e3,
        10e3,
        11e3,
        12e3,
        13e3,
        14e3,
        15e3,
        16e3,
        17e3,
        18e3,
        19e3,
        20e3,
        21e3,
        22e3,
        23e3,
        24e3,
        25e3,
        26e3,
        27e3,
        28e3,
        29e3,
        30e3,
        31e3,
        32e3,
        33e3,
        34e3,
        35e3,
        36e3,
        37e3,
        38e3,
        39e3,
        40e3,
        41e3,
        42e3,
        43e3,
    ]

    true_value = []

    for p in pressures:
        equation_to_solve = (
            lambda r: 3
            * 10**11
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-1.3)
            + 2
            * 10**7
            * (
                (r**2 + outcome["burst_altitude"] ** 2)
                / outcome["burst_energy"] ** (2 / 3)
            )
            ** (-0.57)
            - p
        )
        initial_guess = 1.0
        result = fsolve(equation_to_solve, initial_guess)
        true_value.append(result[0])

    lat, lon, damrad = deepimpact.damage_zones(
        outcome, 55.55, -1.234, 110, pressures
    )

    assert np.isclose(lat, 55.265873883728226)
    assert np.isclose(lon, 0.10096324619319463)

    assert len(true_value) == len(damrad)

    for i in range(len(true_value)):
        assert np.isclose(damrad[i], true_value[i])


def test_radius_value_of_damage_zones(deepimpact):
    outcome = {
        "burst_peak_dedz": 1000.0,
        "burst_altitude": 9000.0,
        "burst_distance": 90000.0,
        "burst_energy": 6000.0,
        "outcome": "Airburst",
    }

    pressures = 27e3

    equation_to_solve = (
        lambda r: 3
        * 10**11
        * (
            (r**2 + outcome["burst_altitude"] ** 2)
            / outcome["burst_energy"] ** (2 / 3)
        )
        ** (-1.3)
        + 2
        * 10**7
        * (
            (r**2 + outcome["burst_altitude"] ** 2)
            / outcome["burst_energy"] ** (2 / 3)
        )
        ** (-0.57)
        - pressures
    )
    initial_guess = 1.0
    result = fsolve(equation_to_solve, initial_guess)
    true_value = result[0]

    lat, lon, damrad = deepimpact.damage_zones(
        outcome, 55.0, 0.0, 135.0, pressures
    )

    assert np.isclose(lat, 54.42365982774329)
    assert np.isclose(lon, 0.983751262482494)

    assert np.isclose(damrad, true_value)


def test_input_of_impact_risk(deepimpact):
    earth = deepimpact.Planet()

    result = earth.solve_atmospheric_entry(
        radius=35, angle=45, strength=1e7, density=3000, velocity=19e3
    )

    result = earth.calculate_energy(result)

    assert deepimpact.impact_risk(earth)
    assert deepimpact.impact_risk(earth, nsamples=1000)
