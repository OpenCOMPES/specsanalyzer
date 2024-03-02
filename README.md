[![Documentation Status](https://github.com/OpenCOMPES/specsanalyzer/actions/workflows/documentation.yml/badge.svg)](https://opencompes.github.io/specsanalyzer/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![](https://github.com/OpenCOMPES/specsanalyzer/actions/workflows/linting.yml/badge.svg)
![](https://github.com/OpenCOMPES/specsanalyzer/actions/workflows/testing_multiversion.yml/badge.svg?branch=main)
![](https://img.shields.io/pypi/pyversions/specsanalyzer)
![](https://img.shields.io/pypi/l/specsanalyzer)
[![](https://img.shields.io/pypi/v/specsanalyzer)](https://pypi.org/project/specsanalyzer)
[![Coverage Status](https://coveralls.io/repos/github/OpenCOMPES/specsanalyzer/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/OpenCOMPES/specsanalyzer?branch=main)

# specsanalyzer
This is the package `specsanalyzer` for conversion and handling of SPECS Phoibos analyzer data.

This package contains two modules:
`specsanalyzer` is a package to import and convert MCP analyzer images from SPECS Phoibos analyzers into energy and emission angle/physical coordinates.
`specsscan` is a Python package for loading Specs Phoibos scans accquired with the labview software developed at FHI/EPFL

Tutorials for usage and the API documentation can be found in the [Documentation](https://opencompes.github.io/specsanalyzer/)

## Installation

### Pip (for users)

- Create a new virtual environment using either venv, pyenv, conda, etc. See below for an example.

```bash
python -m venv .specs-venv
```

- Activate your environment:

```bash
source .specs-venv/bin/activate
```

- Install `specsanalyzer` from PyPI:

```bash
pip install specsanalyzer
```

- This should install all the requirements to run `specsanalyzer` and `specsscan`in your environment.

- If you intend to work with Jupyter notebooks, it is helpful to install a Jupyter kernel for your environment. This can be done, once your environment is activated, by typing:

```bash
python -m ipykernel install --user --name=specs_kernel
```

#### Configuration and calib2d file
The conversion procedures require to set up several configuration parameters in a config file. An example config file is provided as part of the package (see documentation). Configuration files can either be passed to the class constructures, or are read from system-wide or user-defined locations (see documentation).

Most importantly, conversion of analyzer data to energy/angular coordinates requires detector calibration data provided by the manufacturer. The corresponding *.calib2d file (e.g. phoibos150.calbid2d) are provided together with the spectrometer software, and need to be set in the config file.

### For Contributors

To contribute to the development of `specsanalyzer`, you can follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/OpenCOMPES/specsanalyzer.git
cd specsanalyzer
```

2. Check out test data (optional, requires access rights):

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

2. Install the repository in editable mode:

```bash
pip install -e .
```

Now you have the development version of `specsanalyzer` installed in your local environment. Feel free to make changes and submit pull requests.

### Poetry (for maintainers)

- Prerequisites:
  + Poetry: https://python-poetry.org/docs/

- Create a virtual environment by typing:

```bash
poetry shell
```

- A new shell will be spawned with the new environment activated.

- Install the dependencies from the `pyproject.toml` by typing:

```bash
poetry install --with dev, docs
```

- If you wish to use the virtual environment created by Poetry to work in a Jupyter notebook, you first need to install the optional notebook dependencies and then create a Jupyter kernel for that.

  + Install the optional dependencies `ipykernel` and `jupyter`:

  ```bash
  poetry install -E notebook
  ```

  + Make sure to run the command below within your virtual environment (`poetry run` ensures this) by typing:

  ```bash
  poetry run ipython kernel install --user --name=specs_poetry
  ```

  + The new kernel will now be available in your Jupyter kernels list.
