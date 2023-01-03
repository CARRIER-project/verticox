# verticox
This repository contains the components for running vertical cox proportional hazards analysis in a
setting where the data is vertically partitioned.

The solution is based on the Verticox algorithm from
[Dai et al., 2022](https://ieeexplore.ieee.org/document/9076318). It has been adapted to be used
within the [Vantage6](https://vantage6.ai) framework.

This solution will be extended with the scalar vector product protocol to solve certain privacy concerns
in "vanilla" Verticox.

## Current status
The current vantage6 algorithm only implements the original Verticox algorithm. The addition of the 
scalar vector product protocol will be released in the near future. 

## N-party-scalar-product-protocol
We are going to enhance the verticox algorithm by applying the n-party scalar product protocol to the 
components of the verticox algorithm that require querying which samples have a matching event time.

These are the components:

<!--![Selection_180](https://user-images.githubusercontent.com/131889/165753100-6563d7d2-c10e-4a73-93fd-2a77d981e8ab.png) -->

$\sum \limits_{n \in E} \mathbf{x}_{nk}$ (for datanodes)
Where $E$ is the collection of samples that are NOT right-censored.

$\sum \limits_{j \in R_t} exp(K \overline{z}_j)$ (at the central server)

## Other solutions
Besides the fact that we have to take care of component 1 and 2 we also have to perform a sum
based on the event time of the current patient.

## New algorithms
### First-order derivative
In the first order derivative, the part that we have to make privacy-preserving is

$   \sum \limits_{t=1}^{t_u} \left[ d_t \frac{ K exp( Kz_{\overline{u}})}{\sum \limits_{j \in R_t} exp(K \overline{z}_j)} \right] $

We can rewrite this to:

$K exp(Kz_u) \dot \sum \limits_{t=1}^{t_u} \left[ \frac{d_t}{\sum \limits_{j \in R_t} exp(K \overline{z}_j) \right]

We will do this in the following steps
1. Compute $\sum \limits_{j \in R_t} exp(K \overline{z}_j)$ ( component 2 )
2. For every 



## How to use
### Prerequisites
You will need to have the [vantage6](https://vantage6.ai) infrastructure setup to be able to use
this algorithm. Check their website for installation instructions.

### Local installation
You can install the dependencies with pip:
`pip install vantage6-client git+https://github.com/CARRIER-project/verticox.git#subdirectory=python`

### Running the algorithm
You will probably want to check which nodes contain which features before you run the algorithm.
```python
from verticox.client import VerticoxClient
from vantage6.client import Client

# Create a vantage6 client
client = Client(v6_host, v6_port)
client.authenticate(username, password)
client.setup_encryption(private_key)

# Instantiate a verticox client
verticox_client = VerticoxClient(client)

# Get insight into the columns at the datanodes
result = verticox_client.get_column_names()
print(result.get_result())
>> [Result(organization_id=1, content=['gender', 'hr', 'los', 'miord', 'mitype', 'sho', 'sysbp', 'outcome_time', 'outcome']),
>> Result(organization_id=2, content=['outcome_time', 'outcome']),
>> Result(organization_id=3, content=['afb', 'age', 'av3', 'bmi', 'chf', 'cvd', 'diasbp', 'outcome_time', 'outcome'])]
``` 
Now you know which data resides at what node you can run the verticox algorithm:
```python
feature_columns = ['afb', 'age', 'gender']

task = verticox_client.compute(feature_columns, 'outcome_time', 'outcome', [3,5], 2)

# This will take a while
task.get_result()
```

## Python components
Follow the [README](python/README.md) in the `python/` directory.
