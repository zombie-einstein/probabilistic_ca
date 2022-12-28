# Probabilistic Cellular Automata

Python module to run and analyze a probabilistic cellular automata model

See [this blog post](https://zombie-einstein.github.io/2020/06/27/probabilistic_ca.html)
for details of the algorithm and implementation.

## Usage

Examples of using this package and producing plots can be found in the 
jupyter notebook [`usage.ipynb`](usage.ipynb) which uses the module 
[`ca_utils.py`](ca_utils.py).

This has not yet been set up as an installable package, but can be run
from relative imports.

The requirements needed to run the examples in the notebook can be found in
`requirements.txt` and can be installed with `pip install -r requirements.txt`

## JAX Implementation

This project now has a [JAX](https://github.com/google/jax) implementation 
found in [`jax_ca_utils.py`](jax_ca_utils.py) and example notebooks
[`jax_usage.ipynb`](jax_usage.ipynb).
