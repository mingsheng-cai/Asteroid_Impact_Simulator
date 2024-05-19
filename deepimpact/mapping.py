"""This module contains some useful mapping functions"""
import folium
import pandas as pd
import numpy as np

__all__ = ["plot_circle", "MappingFunctions"]


def plot_circle(lat, lon, radius, fmap=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    fmap: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> import deepimpact
    >>> fmap = deepimpact.plot_circle(52.79, -2.95, 1e3, map=None)
    >>> type(fmap)
    <class 'folium.folium.Map'>
    """

    if not fmap:
        fmap = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle(
        [lat, lon], radius, fill=True, fillOpacity=0.6, **kwargs
    ).add_to(fmap)

    return fmap


class MappingFunctions:
    """
    Initializes the MappingFunctions class.

    Attributes:
        earth_radius (float): Earth's radius in kilometers.
        fmap (folium.Map): Folium map instance.
    """

    def __init__(self, fmap=None, lat=55.2, lon=-2.5, data=None):
        self.earth_radius = 6371  # Earth's radius in kilometers
        self.fmap = fmap  # Initialize fmap as an instance variable
        self.lat = lat  # Initialize lat on the map
        self.lon = lon  # Initialize lon on the map
        self.data = data  # Initialize data as an instance variable

    def plot_trajectory(self, start, end, group=None):
        """
        Plot a trajectory line between two points on a map.

        Parameters
        ----------
        start (list): Position of the start point [latitude, longitude].
        end (list): Position of the end point [latitude, longitude].
        group (folium.FeatureGroup, optional): Existing FeatureGroup object.

        Returns
        -------
        None
        """
        if not self.fmap:
            self.fmap = folium.Map(
                location=[start[0], start[1]], control_scale=True
            )

        if not group:
            group = folium.FeatureGroup(name="Trajectories").add_to(self.fmap)

        folium.PolyLine(
            locations=[start, end],
            color="black",
            weight=2,
            tooltip="From Boston to San Francisco",
        ).add_to(group)

    def plot_damage_areas(self, data=None):
        """
        Plot damage areas on a map.

        Parameters
        ----------
        df (dataframe): DataFrame containing the information of damage areas.

        Returns:
            None
        """
        color_map = {1: "lightgreen", 2: "yellow", 3: "orange", 4: "red"}

        if data is None:
            df = self.data
        else:
            df = data

        if not self.fmap:
            initial_location = [df.iloc[0]["lat1"], df.iloc[0]["lon1"]]
            self.fmap = folium.Map(
                location=initial_location, control_scale=True
            )

        feature_groups = []
        for i in range(4):
            feature_groups.append(
                folium.FeatureGroup(name=f"Damage Level {i+1}").add_to(
                    self.fmap
                )
            )
        feature_groups.append(
            folium.FeatureGroup(name="Trajectories").add_to(self.fmap)
        )

        for _, row in df.iterrows():
            self.plot_trajectory(
                [row["lat0"], row["lon0"]],
                [row["lat1"], row["lon1"]],
                feature_groups[4],
            )

            for i in range(4):
                if not pd.isna(row[f"radius{i+1}"]):
                    radius = row[f"radius{i+1}"]
                    folium.Circle(
                        location=[row["lat1"], row["lon1"]],
                        radius=radius,
                        color=color_map.get(i + 1, "black"),
                        stroke=True,
                        fill=True,
                        fill_opacity=0.3,
                        opacity=1,
                        popup=f"Damage Level: {i+1}",
                        tooltip=f"{row[f'radius{i+1}']} radius",
                    ).add_to(feature_groups[i])

        folium.LayerControl().add_to(self.fmap)

    def create_map_df(self, lat, lon, blat, blon, damrad):
        """
        Create a DataFrame containing the information of damage areas.

        Parameters
        ----------
        lat (float): Latitude of the meteoroid entry point (degrees).
        lon (float): Longitude of the meteoroid entry point (degrees).
        blat (float): Latitude of the surface zero point (degrees).
        blon (float): Longitude of the surface zero point (degrees).
        damrad (arraylike, float): List of distances specifying the blast radii
            for the input damage levels.

        Returns:
            dataframe: DataFrame containing the information of damage areas.
        """
        lat, lon = np.array(lat), np.array(lon)
        blat, blon = np.array(blat), np.array(blon)
        damrad = np.sort(np.array(damrad))[::-1]

        data = {
            "lat0": lat,
            "lon0": lon,
            "lat1": blat,
            "lon1": blon,
            "radius1": [None],
            "radius2": [None],
            "radius3": [None],
            "radius4": [None],
        }

        for i, r in enumerate(damrad):
            if r is not None:
                data[f"radius{i+1}"] = [r]

        data = pd.DataFrame(data)
        self.data = data
        return data

    def set_fmap(self, fmap=None):
        """Set the Folium map instance."""
        if fmap is None:
            fmap = folium.Map(location=[self.lat, self.lon], zoom_start=4)
        self.fmap = fmap

    def get_fmap(self):
        """Return the Folium map instance."""
        return self.fmap
