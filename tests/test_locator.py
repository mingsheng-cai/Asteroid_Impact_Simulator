import doctest
import numpy as np
from haversine import haversine, Unit
from pytest import fixture, mark
from deepimpact import locator

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly


@fixture(scope="module")
def deepimpact():
    import deepimpact

    return deepimpact


@fixture(scope="module")
def loc(deepimpact):
    return deepimpact.GeospatialLocator()


def test_import(deepimpact):
    assert deepimpact


class TestGreatCircleDistance(object):
    """Class to test the great_circle_distance function"""

    @mark.parametrize(
        "latlon1, latlon2, expected",
        [
            # Test simple case
            ([[0, 0]], [[0, 0]], 0),
            # Test special case (half equator)
            ([[0, 0]], [180, 0], 6371000 * np.pi),
            # Test random cases
            (
                [[5, 20]],
                [40, 30],
                haversine([5, 20], [40, 30], unit=Unit.METERS),
            ),
            (
                [[-78.26, 35.82]],
                [86.54, 143.22],
                haversine([-78.26, 35.82], [86.54, 143.22], unit=Unit.METERS),
            ),
        ],
    )
    @mark.usefixtures("deepimpact")
    def test_basic_distance(self, deepimpact, latlon1, latlon2, expected):
        # Test for basic functionality
        result = deepimpact.great_circle_distance(latlon1, latlon2)
        assert np.isclose(result, expected, atol=1e-6)

    @mark.parametrize(
        "latlon1, latlon2",
        [
            # Test multiple points
            ([[-10, 0], [0, 20]], [[-30, 160], [20, -90]])
        ],
    )
    @mark.usefixtures("deepimpact")
    def test_multiple_points(self, deepimpact, latlon1, latlon2):
        # Test for multiple points
        result = deepimpact.great_circle_distance(latlon1, latlon2)
        assert result.shape == (2, 2)

    @mark.xfail(reason="Invalid dimensions of inputs")
    @mark.parametrize(
        "latlon1, latlon2",
        [
            # Test invalid inputs with wrong dimensions
            ([[0, 0]], [[-20, 50, 150]])
        ],
    )
    @mark.usefixtures("deepimpact")
    def test_invalid_input(self, deepimpact, latlon1, latlon2):
        result = deepimpact.great_circle_distance(latlon1, latlon2)
        assert result is None


class TestGeospatialLocator(object):
    """Class to test the GeospatialLocator class"""

    @mark.usefixtures("loc")
    def test_get_postcodes_by_radius(self, loc):
        # Test the types of results
        latlon1 = (52.2074, 0.1170)
        result1 = loc.get_postcodes_by_radius(latlon1, [0.2e3, 0.1e3])

        assert type(result1) is list
        if len(result1) > 0:
            for element in result1:
                assert type(element) is list

        # Test for known results
        latlon2 = (51.4981, -0.1773)
        result2 = loc.get_postcodes_by_radius(latlon2, [0.2e3, 0.1e3])

        assert all("SW7 2AZ" in sublist for sublist in result2)

    @mark.xfail(reason="Invalid inputs with negative radii")
    @mark.parametrize(
        "X, radii",
        [
            # Test invalid inputs with negative radii
            ((51.4981, -0.1773), [-0.2e3, 0.1e3])
        ],
    )
    @mark.usefixtures("loc")
    def test_invalid_input_postcodes(self, loc, X, radii):
        result = loc.get_postcodes_by_radius(X, radii)
        assert result is None

    @mark.usefixtures("loc")
    def test_population_by_radius(self, loc):
        # Test the types of results
        latlon1 = (52.2074, 0.1170)
        result1 = loc.get_population_by_radius(latlon1, [5e2, 1e3])

        assert type(result1) is list
        if len(result1) > 0:
            for element in result1:
                assert type(element) is int

        # Test for simple condition with ouput of 0
        latlon2 = (60.7505, 26.4321)
        result2 = loc.get_population_by_radius(latlon2, [1e2, 5e2, 1e3])

        assert np.equal(result2, 0).all()

    @mark.xfail(reason="Invalid inputs with negative radii")
    @mark.parametrize(
        "X, radii",
        [
            # Test invalid inputs with negative radii
            ((52.2074, 0.1170), [1e2, -5e2, 1e3])
        ],
    )
    @mark.usefixtures("loc")
    def test_invalid_input_population(self, loc, X, radii):
        result = loc.get_postcodes_by_radius(X, radii)
        assert result is None


class TestDocstrings(object):
    """Class to test the docstrings"""

    def test_postcodes_docstrings(self):
        """Test the docstrings in locator module"""
        result = doctest.testmod(locator)
        assert result.failed == 0, f"{result.failed} docstring test(s) failed"
