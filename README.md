# ACS-1-deepimpact

### Synopsis

This project explores the impact of small asteroids entering Earth's atmosphere. These space rocks, when encountering the Earth, undergo extreme forces that result in deceleration, heating, and potential disruption. The fate of an asteroid is influenced by various factors including its mass, speed, trajectory angle, and internal strength.

Asteroids with diameters ranging from 10-100 meters can penetrate deep into the Earth’s atmosphere and cause catastrophic disruptions, leading to atmospheric disturbances known as airbursts. These events can have significant ground-level impacts, as exemplified by the Chelyabinsk event in 2013 and the Tunguska event in 1908.

### Objective

The primary objective of this project is to analyze and understand the effects and hazards posed by small asteroids upon entering the Earth’s atmosphere. The study involves examining historical events, modeling asteroid interactions with the Earth's atmosphere, and assessing potential ground-level impacts.

## Getting Started

### Installatation

```bash
# Create the `deepimpact` environment
conda env create -f environment.yml

# Activate the `deepimpact` environment
conda activate deepimpact

# Install the `deepimpact` package
pip install -e .

# Install the `deepimpact` dependencies
pip install -r requirements.txt
```

To run the GUI program `examples/example_mapping_ui.py` please install the following package

```shell
pip uninstall PyQt5 PyQtWebEngine
pip install PyQt5 PyQtWebEngine
```

### Downloading postcode data

To download the postcode data
```
python download_data.py
```

### Automated testing

To run the pytest test suite, from the base directory run
```
pytest tests/
```

### Documentation

To generate the documentation (in html format)
```
python -m sphinx docs html
```


### Example usage

For example usage see `example.py` in the examples folder:
```
python examples/example.py
```

### Performance Benchmarking

For more information on the project performance, see the notebook `solver_analysis.ipnyb` in the analysis folder

## Contributing

We value your contributions to ACS-1 Deep Impact! To keep everything organized and efficient, please follow these brief guidelines:
- Keep contributions clear and concise.
- Follow coding standards and best practices.
- Be respectful and open to feedback.

#### Automatically formatting your python code:

```
python3 -m pip install black
black .
```

## More information

For more information on the project specfication, see the notebooks: `ProjectDescription.ipynb`, `AirburstSolver.ipynb` and `DamageMapper.ipynb`.


## References

See the references.md file

## License

This project is released under the [MIT](LICENSE).
