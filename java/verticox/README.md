## Verticox N-party scalar product protocol wrapper

This is a wrapper that can be used to incorperate the N-party scalar product protocol within verticox by Dai et al. The
advantage of using this wrapper is that you no longer need to share the time of events among all parties. This wrapper
allows you to set a value per individual. Then insert this value into the n-party protocol.

The assumption is that only 1 party contains values (e.g. co-variates) for a given value the other parties only contain
selection criteria (e.g. individual is right-cencored, individual has event-time t ). It is possible for party A to
contain values for individual 1, party B for individual 2 and party C knows the selection criteria. It is also
acceptable if A knows both the relevant values & selection criteria. But it should not be the case that both A & B know
relevant values

To use this method the following needs to be done:

1) setPrecision (default 5)
2) determineMinimumPeriod (can be skipped if you minimumperiods are already known)
3) setValues for the party who owns the relevant data
4) sumRelevantValues

## Implemented methods:

### setPrecision:

Set the precision to be used for double values for the product protocol. Expected input:

- An int indicating the precision

Default precision used is 5. Always make sure to keep the precision the same across the various parties involved. To
keep everything alligned use setPrecisionCentral

### getPublicKey:

A method to acces the public key used in the RSA protocol. This allows us to efficiently encrypt the auxilery values
using the AES protocol. The AES key should then be encrypted using this key. This protects against a man in the middle
attack. Important to note: this encryption is needed due to the split code between 2 docker images for our verticox
implementation in Vantage6. This split nature requires us to communicate the auxilery values between the two docker
images 1 party owns. This communication runs via a VPN that is maintained within the vantage6 infrastructure. An
implementation of the verticox protocol that does not utilize this split infrastructure would not need to encrypt the
auxilery values as it does not need to communicate them between two docker images.

### setValues:

A method to set the auxilery values for each individual. Expected input:

- List of auxilery values, ordered according to the shared population ordering. These values are encrypted usin AES
- Encrypted AES key using the RSA publickey provided by the webservice

### initValues:

A method that inits the localdata for the scalar product protocol with locally available values. Expected input:

- List of selection criteria, for example the time period of relevant individuals.

### determineMinimumPeriod:

A method that can determine the smallest time period needed to include at least 10 individuals based on a given starting
point. Expected input:

- Starting time represented as an Attribute-value for the attribute representing the time of events.

To avoid having to call every single datastation yourself use determineMinimumPeriodCentral

### selectRelevantIndividuals:

Select a population to be used in the scalar product protocol according to certain criteria. This method does not need
to be called independently. Expected input:

- List of selection criteria, for example the time period of relevant individuals

### sumRelevantValues:

Sums the values based on the relevant individuals. Expected input:

- List of selection criteria, for example the time period of relevant individuals

### Handling a Hybird split in the data

To handle a Hybrid split in your data include an attributecolumn in all relevant datasets named "locallyPresent" with "
bool" as it's type. Locally available data should have the value "TRUE". Missing records are then inserted as a row that
has the value "FALSE" for this attribute. This should be handled in a preprocessing step.

Important to note; datasets still need to have the same ordering for their records. It is assumed that recordlinkage is
handled in a preprocessing step as well.

This functionality is only available in the java implementation.