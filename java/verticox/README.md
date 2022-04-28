## Verticox N-party scalar product protocol wrapper

This is a wrapper that can be used to incorperate the N-party scalar product protocol within verticox by Dai et al. The
advantage of using this wrapper is that you no longer need to share the time of events among all parties.

### ToDo:

Local data is currently not actually used. This seems wrong.

The following methods are implemented:

### setPrecision:

Set the precision to be used for double values for the product protocol. Expected input:

- An int indicating the precision

### updateCovariateVariables:

A method to update the locally available covariate values. Expected input:

- Attributename of relevant attribute
- List of covariate values of this attribute, ordered according to the shared population ordering

### initCovariateVariables:

A method that inits the localdata for the scalar product protocol with locally available covariate. Expected input:

- Attributename of relevant attribute

### setZValues:

A method to set the auxilery values for each individual. Expected input:

- List of z-values, ordered according to the shared population ordering

### initZValues:

A method that inits the localdata for the scalar product protocol with locally available zValues. Expected input:

### determineMinimumPeriod:

A method that can determine the smallest time period needed to include at least 10 individuals based on a given starting
point. Expected input:

- Starting time represented as an Attribute-value for the attribute representing the time of events.

### selectRelevantIndividuals:

Select a population to be used in the scalar product protocol according to certain criteria. Expected input:

- List of selection criteria, for example the time period of relevant individuals

