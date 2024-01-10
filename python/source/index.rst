.. verticox+ documentation master file, created by
sphinx-quickstart on Mon Oct 16 15:20:35 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to verticox+'s documentation!
=====================================
This repository contains the components for running vertical cox proportional hazards analysis in a
setting where the data is vertically partitioned.

The solution is based on the Verticox algorithm from
`Dai et al., 2022 <https://ieeexplore.ieee.org/document/9076318)>`_. It has been adapted to be used
within the `Vantage6 <https://vantage6.ai>`_ framework.

This solution is extended with the scalar vector product protocol to solve certain privacy
concerns in "vanilla" Verticox.

The original verticox algorithm is as follows:

Algorithm 1 Distributed Cox Model for Vertically Partitioned Data
-----------------------------------------------------------------

1. **Initialization:** :math:`p = 1`, :math:`\beta_k^{(0)} = 0`, :math:`\bar{z}_n^{(0)} = 0`, and :math:`\gamma_{nk}^{(0)} = 0`
2. **while** :math:`||\bar{z}^{(p)} - \bar{z}^{(p-1)}||_F > \epsilon` **and** :math:`||\bar{z}^{(p)} - \sigma^{(p)}||_F > \epsilon` **do**
3.   - **for** institution :math:`k = 1, \ldots, K` **do**
4.       - Institution :math:`k` solves :math:`\beta_k^{(p)}` using Eq. (8)
5.       - Institution :math:`k` sends :math:`\gamma_{nk}^{(p)}` and the aggregated result :math:`\sigma_{nk}^{(p)}` to the server
6.   - **end for**
7.   - Server computes :math:`\bar{\sigma}_n^{(p)} = \sum_{k=1}^{K} \sigma_{nk}^{(p)} / K` and :math:`\bar{\gamma}_n^{(p)} = \sum_{k=1}^{K} \gamma_{nk}^{(p)} / K`
8.   - Server calculates :math:`\bar{z}^{(p)}` by solving Eq. (12) with Newton-Raphson method
9.   - **Initialization:** :math:`\bar{z}^{(p,0)} = \bar{z}^{(p-1)}`, :math:`q = 1`
10.  - **while** :math:`||\bar{z}^{(p,q)} - \bar{z}^{(p,q-1)}||_F > \epsilon` **do**
11.      - Server calculates :math:`\frac{dL_{z}(\bar{z})}{dz}` at :math:`\bar{z}^{(p,q-1)}` using Eq. (13)
12.      - Server calculates :math:`\frac{d^2L_{z}(\bar{z})}{dz^2}` at :math:`\bar{z}^{(p,q-1)}` using Eq. (14)
13.      - Server updates :math:`\bar{z}^{(p,q)}` using Eq. (16)
14.  - **end while**
15.  - Server updates :math:`z_{nk}^{(p)}` using Eq. (11)
16.  - Server sends :math:`z_{nk}^{(p)}`, :math:`\bar{\sigma}_n^{(p)}`, and :math:`\bar{\gamma}_n^{(p)}` to corresponding institution :math:`k = 1, \ldots, K`
17.  - **for** institution :math:`k = 1, \ldots, K` **do**
18.      - Institution :math:`k` updates :math:`\gamma_{nk}^{(p)}` using Eq. (18)
19.  - **end for**
20. **end while**

The main privacy issue lies within solving :math:`\beta_k^{(p)}` at the institutions.
:math:`\beta_k^{(p)}` is solved with the following equation:

.. math::

    \beta_k^{(p)} = \left[ \rho \sum_{n=1}^{N} x_{nk}^T x_{nk} \right]^{-1}
    \cdot
    \left[ \sum_{n=1}^{N} \left( \rho z_{nk}^{(p-1)} - \gamma_{nk}^{(p-1)} \right) x_{nk}^T
    + \sum_{t=1}^{T} \sum_{n \in D_t} x_{nk} \right]

Let us focus on the last part of the equation here:

.. math::

   \sum_{t=1}^{T} \sum_{n \in D_t} x_{nk}

Here, there is a reference to :math:`\mathcal{D}_t`, which is the index set of samples with an
observed
event at time :math:`t`. So for every time :math:`t` we need to select the samples with an observed
event. This requires the availability of outcome data at every institution. In real-world usecases
this is not always possible.

Verticox+ solves this problem by making use of the n-party-scalar-product-protocol.
In order to do that, we translate the inner sum :math:`\sum_{n \in D_t} x_nk` to a scalar product:


.. math::
      u_{kt} = x_k \cdot \vec{D_t}

In this case, :math:`\vec{D_t}` is the boolean vector of length :math:`N` that indicates for each
sample whether it had an event at time :math:`t` or not.

Since :math:`u_{kt}` per time :math:`t` stays constant over iterations. We will only compute this at
the beginning.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
