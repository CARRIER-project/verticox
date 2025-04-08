[![Build and Publish](https://github.com/CARRIER-project/verticox/actions/workflows/push.yml/badge.svg)](https://github.com/CARRIER-project/verticox/actions/workflows/push.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13933626.svg)](https://doi.org/10.5281/zenodo.13933626)

[Documentation](https://carrier-project.github.io/verticox/)

Authors:

- Djura Smits: <d.smits@esciencecenter.nl>
- Florian van Daalen: <f.vandaalen@maastrichtuniversity.nl>

# verticox+
This repository contains the components for running vertical cox proportional hazards analysis in a
setting where the data is vertically partitioned.

The solution is based on the Verticox algorithm from
[Dai et al., 2022](https://ieeexplore.ieee.org/document/9076318). It has been adapted to be used
within the [Vantage6](https://vantage6.ai) framework.

This solution is extended with the scalar vector product protocol to remove the requirement that
outcome data needs to be available at all parties.

