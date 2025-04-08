[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13933626.svg)](https://doi.org/10.5281/zenodo.13933626)

# Introduction

Verticox+ is a Cox proportional hazards algorithm for vertically distributed data.

The solution is based on the Verticox algorithm from
[Dai et al., 2022](https://ieeexplore.ieee.org/document/9076318). It has been adapted to be used
within the [Vantage6](https://vantage6.ai) framework.

This solution is extended with the scalar vector product protocol
([van Daalen et al.](https://doi.org/10.48550/arXiv.2112.09436)) to solve certain privacyconcerns in
"vanilla" Verticox.

## Quickstart
The main way to use verticox+ is to run it within a vantage6 environment. For more info on vantage6, 
see the [vantage6 documentation](https://docs.vantage6.ai/).

When you have a vantage6 environment running either locally or remotely, you can test out verticox
by running the [demo](demo.md).

## Theory

The original Verticox pseudocode can be summarized as below.
The full pseudocode can be found in the original paper by Dai *et al.*

### Algorithm 1: Original Verticox Algorithm

**Input:** Local data at each party  
**Output:** Converged Cox proportional hazard model

1. **Initialization**
2. **While** stopping criterion has not been reached:
    1. **For** each party \( k \):
        - Solve \( \beta_{k}^{p} \)
        - Compute \( \sigma_{nk} = \beta_k^T x_{nk} \)
        - Send \( \sigma_{nk} \) to the central server
    2. Server aggregates subresults
    3. Server calculates auxiliary value \( \overline{z^{p}} \)
    4. Server updates \( z_{nk}^{p} \)
    5. Server sends \( z_{nk}^{p} \) and aggregation to parties
    6. Local parameters are updated

The main privacy issue lies within solving \( \beta_{k}^{p} \). This is done using the following
equation:

\[
\beta_{k}^{p} = \left[ \rho \sum_{n=1}^{N} x_{nk}^{T}x_{nk} \right]^{-1} \cdot
\left[ \sum_{n=1}^{N} (\rho z_{nk}^{p-1} - \gamma_{nk}^{p-1} ) x_{nk}^{T} + \sum_{t=1}^{T}\sum_{n\in E_{t}}x_{nk} \right]
\]

The problem lies in the last part of the equation: \( \sum_{t=1}^{T}\sum_{n\in E_{t}}x_{nk} \). This
part references \( E_{t} \), which is the index set of samples with an observed event at time \(
t \). Therefore, for every time \( t \), we need to select the samples with an observed event. This
requires the availability of outcome data at every party, which is not always possible in real-world
use cases.

Verticox+ will solve this problem by using the scalar-product protocol. To do that, we translate the
inner sum \( \sum_{n\in E_{t}}x_{nk} \) to a scalar product:

\[
u_{kt} = x_{k} \cdot \overrightarrow{(E_{t})}
\]

In this case, \( \overrightarrow{(E_{t})} \) is a Boolean vector of length \( N \) that indicates
for each sample whether it had an event at time \( t \) (indicated as 1) or not (indicated as 0). \(
\beta_k^{p} \) will now be solved according to the following equation:

\[
\beta_{k}^{p} = \left[ \rho \sum_{n=1}^{N} x_{tnk}^{T}x_{tnk} \right]^{-1}
\cdot
\left[ \sum_{n=1}^{N} (\rho z_{nk}^{p-1} - \gamma_{nk}^{p-1} ) x_{nk}^{T} + \sum_{t=1}^{T} u_{kt} \right]
\]

Authors:

- Djura Smits <d.smits@esciencecenter.nl>
- Florian van Daalen <f.vandaalen@maastrichtuniversity.nl>