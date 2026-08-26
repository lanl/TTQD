
# TTNQD:

[`ttnqd`](https://github.com/ares201005/ttnqd): Tree Tensor Network Quantum Dynamics (TTNQD) package **N** means "Network", "Nuclear", "N-dimemnsional", "Nonequilibrium" (you name it).

-----------------------------------------------


## O5131

Copyright 2026. Triad National Security, LLC. All rights reserved.

## Authors:

[Yu Zhang](mailto:zhy@lanl.gov)

## Features:

TBA

## Citation:

TBA

## Installation


### Install lib

(c++ lib TBA)

```
  mkdir build && cd build
  cmake ../
  make
  make install
```

### setup python code

```
export PYTHONPATH=$PYTHONPATH:/path/to/ttqd

```

## Dependencies

Required packages, tools, and libs:

TBA

## Optional packages

TBA

## Tests

The Python tests badge at the top of this README shows the current status of
the GitHub Actions CI workflow.

The CI workflow uses `pytest` and runs the stable numbered test suite `01`
through `06` on pushes and pull requests to `main` and `develop`:

```
python -m pytest
```

The selected files are configured in `pytest.ini`:

```
tests/01_tensors.py
tests/02_ttn.py
tests/03_networktools.py
tests/04_dmrg_mps.py
tests/05_dmrg_ttn.py
tests/06_tdvp.py
```

Additional tests in `tests/` can be run explicitly by path when needed. The
legacy unittest runner remains available as `python tests/run_tests.py`.

## Documentation

Details of installation instruction, usage, APIs, and examples can be
found in https://ttqd.readthedocs.io/en/latest/ (TBA) or from local builds

html:
```
  cd docs
  make html
```

pdf:
```
  make latexpdf
```
