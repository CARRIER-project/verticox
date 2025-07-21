[![Build and Publish](https://github.com/CARRIER-project/verticox/actions/workflows/push.yml/badge.svg)](https://github.com/CARRIER-project/verticox/actions/workflows/push.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13933626.svg)](https://doi.org/10.5281/zenodo.13933626)

[Documentation](https://carrier-project.github.io/verticox/)

Authors:

- Djura Smits: <d.smits@esciencecenter.nl>
- Florian van Daalen: <f.vandaalen@maastrichtuniversity.nl>

# verticox+
This repository contains the components for running vertical cox proportional hazards analysis in a
setting where the data is vertically partitioned.

Our solution Verticox+[[1]](#1) is based on the Verticox algorithm by Dai et al. [[2]](#2). It has been adapted to be used
within the [Vantage6](https://vantage6.ai) framework. 

This solution is extended with the scalar vector product protocol to remove the requirement that
outcome data needs to be available at all parties.

## References

- <a id="1">[1]</a> van Daalen, F., Smits, D., Ippel, L. et al. Verticox+: vertically distributed Cox proportional hazards model with improved privacy guarantees. Complex Intell. Syst. 11, 388 (2025), https://doi.org/10.1007/s40747-025-02022-4
- <a id="2">[2]</a> W. Dai, X. Jiang, L. Bonomi, Y. Li, H. Xiong and L. Ohno-Machado, "VERTICOX: Vertically Distributed Cox Proportional Hazards Model Using the Alternating Direction Method of Multipliers," in IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 2, pp. 996-1010, 1 Feb. 2022, doi: https://10.1109/TKDE.2020.2989301.
