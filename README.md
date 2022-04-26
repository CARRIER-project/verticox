[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/CARRIER-project/verticox/HEAD?labpath=Verticox.ipynb)

# verticox

# N-party-scalar-product-protocol
We are going to enhance the verticox algorithm by applying the n-party scalar product protocol to the 
components of the verticox algorithm that require querying which samples have a matching event time.

These are the components:

## $\sum \limits_{n \in E} \mathbf{x}_{nk}$ (for datanodes)$
Where $E$ is the collection of samples that are NOT right-censored.

## $\sum \limits_{j \in R_t} exp(K \overline{z}_j)$