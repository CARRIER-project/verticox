# Welcome to verticox+'s documentation!
This repository contains the components for running vertical cox proportional hazards analysis in a
setting where the data is vertically partitioned.

The solution is based on the Verticox algorithm from
[Dai et al., 2022](https://ieeexplore.ieee.org/document/9076318). It has been adapted to be used
within the [Vantage6](https://vantage6.ai) framework.

This solution is extended with the scalar vector product protocol to solve certain privacy
concerns in "vanilla" Verticox.

The original verticox algorithm is as follows:

## Algorithm 1 Distributed Cox Model for Vertically Partitioned Data

1. **Initialization:** \( p = 1 \), \( \beta_k^{(0)} = 0 \), \( \bar{z}_n^{(0)} = 0 \), and \( \gamma_{nk}^{(0)} = 0 \)
2. **while** \( ||\bar{z}^{(p)} - \bar{z}^{(p-1)}||_F > \epsilon \) **and** \( ||\bar{z}^{(p)} - \sigma^{(p)}||_F > \epsilon \) **do**
3.   - **for** institution \( k = 1, \ldots, K \) **do**
4.       - Institution \( k \) solves \( \beta_k^{(p)} \) using Eq. (8)
5.       - Institution \( k \) sends \( \gamma_{nk}^{(p)} \) and the aggregated result \( \sigma_{nk}^{(p)} \) to the server
6.   - **end for**
7.   - Server computes \( \bar{\sigma}_n^{(p)} = \sum_{k=1}^{K} \sigma_{nk}^{(p)} / K \) and \( \bar{\gamma}_n^{(p)} = \sum_{k=1}^{K} \gamma_{nk}^{(p)} / K \)
8.   - Server calculates \( \bar{z}^{(p)} \) by solving Eq. (12) with Newton-Raphson method
9.   - **Initialization:** \( \bar{z}^{(p,0)} = \bar{z}^{(p-1)} \), \( q = 1 \)
10.  - **while** \( ||\bar{z}^{(p,q)} - \bar{z}^{(p,q-1)}||_F > \epsilon \) **do**
11.      - Server calculates \( \frac{dL_{z}(\bar{z})}{dz} \) at \( \bar{z}^{(p,q-1)} \) using Eq. (13)
12.      - Server calculates \( \frac{d^2L_{z}(\bar{z})}{dz^2} \) at \( \bar{z}^{(p,q-1)} \) using Eq. (14)
13.      - Server updates \( \bar{z}^{(p,q)} \) using Eq. (16)
14.  - **end while**
15.  - Server updates \( z_{nk}^{(p)} \) using Eq. (11)
16.  - Server sends \( z_{nk}^{(p)} \), \( \bar{\sigma}_n^{(p)} \), and \( \bar{\gamma}_n^{(p)} \) to corresponding institution \( k = 1, \ldots, K \)
17.  - **for** institution \( k = 1, \ldots, K \) **do**
18.      - Institution \( k \) updates \( \gamma_{nk}^{(p)} \) using Eq. (18)
19.  - **end for**
20. **end while**

The main privacy issue lies within solving \( \beta_k^{(p)} \) at the institutions.
\( \beta_k^{(p)} \) is solved with the following equation:

\[ \beta_k^{(p)} = \left[ \rho \sum_{n=1}^{N} x_{nk}^T x_{nk} \right]^{-1} \cdot \left[ \sum_{n=1}^{N} \left( \rho z_{nk}^{(p-1)} - \gamma_{nk}^{(p-1)} \right) x_{nk}^T + \sum_{t=1}^{T} \sum_{n \in D_t} x_{nk} \right] \]

Let us focus on the last part of the equation here:

\[ \sum_{t=1}^{T} \sum_{n \in D_t} x_{nk} \]

Here, there is a reference to \( \mathcal{D}_t \), which is the index set of samples with an
observed event at time \( t \). So for every time \( t \) we need to select the samples with an observed
event. This requires the availability of outcome data at every institution. In real-world use cases
this is not always possible.

Verticox+ solves this problem by making use of the n-party-scalar-product-protocol.
In order to do that, we translate the inner sum \( \sum_{n \in D_t} x_nk \) to a scalar product:

\[ u_{kt} = x_k \cdot \vec{D_t} \]

In this case, \( \vec{D_t} \) is the boolean vector of length \( N \) that indicates for each
sample whether it had an event at time \( t \) or not.

Since \( u_{kt} \) per time \( t \) stays constant over iterations. We will only compute this at
the beginning.
