import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QWidget, QLineEdit, QSlider, QLabel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt
import io
import deepimpact


class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Map")
        self.setGeometry(100, 100, 1200, 800)  # Adjusted layout
        self.earth = deepimpact.Planet()
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout is horizontal
        main_layout = QHBoxLayout(central_widget)

        # Control panel with a vertical layout
        control_layout = QVBoxLayout()
        # Input fields for lat, lon
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("Enter Latitude")
        control_layout.addWidget(self.lat_input)
        self.lat_input.textChanged.connect(self.update_map)

        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("Enter Longitude")
        control_layout.addWidget(self.lon_input)
        self.lon_input.textChanged.connect(self.update_map)

        # Slider for radius
        self.radius = QSlider(Qt.Horizontal)
        self.radius.setMinimum(0)  # Minimum radius
        self.radius.setMaximum(200)  # Maximum radius
        self.radius.setValue(35)  # Default radius
        control_layout.addWidget(self.radius)
        self.radius_label = QLabel(f"Radius: {self.radius.value()}")
        control_layout.addWidget(self.radius_label)
        self.radius.valueChanged[int].connect(self.onSliderValueChanged)

        # Slider for angle
        self.angle = QSlider(Qt.Horizontal)
        self.angle.setMinimum(0)
        self.angle.setMaximum(90)
        self.angle.setValue(45)
        control_layout.addWidget(self.angle)
        self.angle_label = QLabel(f"Angle: {self.angle.value()}")
        control_layout.addWidget(self.angle_label)
        self.angle.valueChanged[int].connect(self.onSliderValueChanged)

        # Slider for strength
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setMinimum(0)
        self.strength.setMaximum(10000000)
        self.strength.setValue(1000000)
        control_layout.addWidget(self.strength)
        self.strength_label = QLabel(f"Strength: {self.strength.value()}")
        control_layout.addWidget(self.strength_label)
        self.strength.valueChanged[int].connect(self.onSliderValueChanged)

        # Slider for density
        self.density = QSlider(Qt.Horizontal)
        self.density.setMinimum(0)
        self.density.setMaximum(5000)
        self.density.setValue(3000)
        control_layout.addWidget(self.density)
        self.density_label = QLabel(f"Density: {self.density.value()}")
        control_layout.addWidget(self.density_label)
        self.density.valueChanged[int].connect(self.onSliderValueChanged)

        # Slider for velocity
        self.velocity = QSlider(Qt.Horizontal)
        self.velocity.setMinimum(0)
        self.velocity.setMaximum(30000)
        self.velocity.setValue(19000)
        control_layout.addWidget(self.velocity)
        self.velocity_label = QLabel(f"Velocity: {self.velocity.value()}")
        control_layout.addWidget(self.velocity_label)
        self.velocity.valueChanged[int].connect(self.onSliderValueChanged)

        # Slider for bearing
        self.bearing = QSlider(Qt.Horizontal)
        self.bearing.setMinimum(0)
        self.bearing.setMaximum(360)
        self.bearing.setValue(217)
        control_layout.addWidget(self.bearing)
        self.bearing_label = QLabel(f"Bearing: {self.bearing.value()}")
        control_layout.addWidget(self.bearing_label)
        self.bearing.valueChanged[int].connect(self.onSliderValueChanged)

        # Add control layout to the main layout
        main_layout.addLayout(control_layout)

        # Map display
        self.map_view = QWebEngineView()
        main_layout.addWidget(self.map_view)

        # Initial map
        self.update_map()

    def onSliderValueChanged(self, value):
        # Update the label of the slider that triggered this event
        sender = self.sender()
        if sender == self.radius:
            self.radius_label.setText(f"Radius: {value}")
        elif sender == self.angle:
            self.angle_label.setText(f"Angle: {value}")
        elif sender == self.strength:
            self.strength_label.setText(f"Strength: {value}")
        elif sender == self.density:
            self.density_label.setText(f"Density: {value}")
        elif sender == self.velocity:
            self.velocity_label.setText(f"Velocity: {value}")
        elif sender == self.bearing:
            self.bearing_label.setText(f"Bearing: {value}")
        self.update_map()

    def update_map(self):
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
        except ValueError:
            lat, lon = 55.2, -2.5  # Default values

        radius = self.radius.value()
        angle = self.angle.value()
        strength = self.strength.value()
        density = self.density.value()
        velocity = self.velocity.value()
        bearing = self.bearing.value()
        # Solve the atmospheric entry problem
        result = self.earth.solve_atmospheric_entry(
            radius=radius,
            angle=angle,
            strength=strength,
            density=density,
            velocity=velocity,
        )
        # Calculate the kinetic energy lost per unit altitude and add it
        result = self.earth.calculate_energy(result)
        # Determine the outcomes of the impact event
        outcome = self.earth.analyse_outcome(result)
        # Pressure levels
        pressures = [1e3, 4e3, 30e3, 50e3]

        blast_lat, blast_lon, damage_rad = deepimpact.damage_zones(
            outcome, lat=lat, lon=lon, bearing=bearing, pressures=pressures
        )
        mf = deepimpact.mapping.MappingFunctions()
        df = mf.create_map_df(lat, lon, blast_lat, blast_lon, damage_rad)
        mf.plot_damage_areas(df)
        fmap = mf.get_fmap()

        data = io.BytesIO()
        fmap.save(data, close_file=False)
        self.map_view.setHtml(data.getvalue().decode())


def main():
    app = QApplication(sys.argv)
    main_window = MapWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
