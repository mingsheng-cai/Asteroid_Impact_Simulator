"""Module dealing with postcode information."""

import numpy as np
import pandas as pd
import os

__all__ = ["GeospatialLocator", "great_circle_distance"]


def great_circle_distance(latlon1, latlon2):
    """Calculation of great circle distance between point pairs.

    Parameters
    ----------
    latlon1 : arraylike
        Latitudes and longitudes of the first
        point (as [n, 2] array for n points).
    latlon2 : arraylike
        Latitudes and longitudes of the second point
        (as [m, 2] array for m points or [2] for a single point).

    Returns
    -------
    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array).

    Examples
    --------
    >>> print(great_circle_distance([[0, 0], [20, 0]], [180, 0]))
    [[20015086.79602057]
     [17791188.26312939]]
    """

    # Earth's radius in meters
    R = 6371000.0

    # Convert input parameters to radians
    latlon1 = np.radians(latlon1)
    latlon2 = np.radians(latlon2)

    # Ensure latlon2 is 2-dimensional
    if latlon2.ndim == 1:
        latlon2 = latlon2.reshape(1, -1)

    # Split latitudes and longitudes
    lat1, lon1 = latlon1[:, np.newaxis, 0], latlon1[:, np.newaxis, 1]
    lat2, lon2 = latlon2[np.newaxis, :, 0], latlon2[np.newaxis, :, 1]

    # Calculate differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance
    distance = R * c

    return distance


class GeospatialLocator(object):
    """Interact with a postcode database file and a population grid file.

    Class to interact with a postcode database file and a population grid file.
    """

    def __init__(
        self,
        postcode_file=os.sep.join(
            (
                os.path.dirname(__file__),
                "..",
                "resources",
                "full_postcodes.csv",
            )
        ),
        census_file=os.sep.join(
            (
                os.path.dirname(__file__),
                "..",
                "resources",
                "UK_residential_population_2011_latlon.asc",
            )
        ),
        norm=great_circle_distance,
        grid_size=1000,
    ):
        """
        Parameters:
        ----------
        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .asc file containing census data on a
            latitude-longitude grid.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """
        self.norm = norm
        self.postcodes = pd.read_csv(postcode_file)
        self.get_population_data(census_file)
        self.grid_size = grid_size

    def get_postcodes_by_radius(self, X, radii):
        """Return postcodes within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.


        Examples
        --------
        >>> locator = GeospatialLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.1e3])
        [['SW7 2AZ']]
        """

        # Create empty list to store postcodes
        postcodes_list = []

        # Calculate the distance from the input location to each postcode
        self.postcodes["Distance"] = self.norm(
            self.postcodes[["Latitude", "Longitude"]].values, X
        )

        # Loop over the radii and append the postcodes within each radius
        for radius in radii:
            postcodes_list.append(
                self.postcodes[self.postcodes["Distance"] < radius][
                    "Postcode"
                ].values.tolist()
            )

        return postcodes_list

    def get_population_data(self, file_path):
        """
        Get population data from the .asc file.

        Parameters
        ----------
        file_path : str
            Path to the .asc file.

        Returns
        -------
        None
        """
        with open(file_path, "r") as file:
            lines = file.readlines()

        nrows = int(lines[1].split()[1])
        nodata_value = int(lines[2].split()[1])

        # Split data into latitude, longitude, and population
        data = [
            list(map(float, lines[i].split())) for i in range(6, len(lines))
        ]
        latitude = np.array(data[:nrows]).flatten()
        longitude = np.array(data[nrows: 2 * nrows]).flatten()
        population = np.array(data[2 * nrows:]).flatten()
        population = np.where(population == nodata_value, 0, population)

        # Create a DataFrame
        self.population_df = pd.DataFrame(
            {
                "Latitude": latitude,
                "Longitude": longitude,
                "Population": population,
            }
        )

    def get_population_by_radius(self, X, radii):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list
            Contains the population closer than the elements of radii to
            the location X. Output should be the same shape as the radii array.

        Examples
        --------
        >>> locator = GeospatialLocator()
        >>> locator.get_population_by_radius((51.4981, -0.1773),\
            [1e2, 5e2, 1e3])
        [0, 7412, 27794]
        """

        population_list = []

        # Calculate the distance from the input location to each grid cell
        self.population_df["Distance"] = self.norm(
            self.population_df[["Latitude", "Longitude"]].values, X
        )

        # Loop over the radii and append the postcodes within each radius
        for radius in radii:
            total_population = self.population_df[
                self.population_df["Distance"] < radius
            ]["Population"].sum()
            population_list.append(int(total_population))

        return population_list

    def get_population_by_radius_optimized(self, X, radii):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list
            Contains the population closer than the elements of radii to
            the location X. Output should be the same shape as the radii array.

        Examples
        --------
        >>> locator = GeospatialLocator()
        >>> locator.get_population_by_radius_optimized((51.4981, -0.1773),\
            [1e2, 5e2, 1e3])
        [232, 7412, 27794]
        """
        population_list = []

        for radius in radii:
            # Calculate the distance from the input location to each grid cell
            self.population_df["Distance"] = self.norm(
                self.population_df[["Latitude", "Longitude"]].values, X
            )
            nearest_grids = self.population_df[
                self.population_df["Distance"] <= radius
            ]

            if nearest_grids.empty:
                # If no grids are close enough,
                # return an estimate based on the nearest grid
                nearest_grid = self.population_df.iloc[
                    self.population_df["Distance"].argmin()
                ]
                population_density = nearest_grid["Population"] / (
                    self.grid_size**2
                )
                circle_area = np.pi * radius**2
                population_list.append(int(population_density * circle_area))
            else:
                # If there are grids close enough,
                # add up the population of all the grids included
                # Loop over the radii and append the
                # postcodes within each radius
                total_population = nearest_grids["Population"].sum()
                population_list.append(int(total_population))

        return population_list
