## Verticox N-party scalar product protocol wrapper

This is a wrapper that can be used to incorperate the N-party scalar product protocol within verticox by Dai et al. The
advantage of using this wrapper is that you no longer need to share the outcome among all parties. Instead this wrapper
will calculate the value in a privacy preserving manner. See associated powerpoint in this repositories for a more
detailed explanation.

It is possible for party A to contain values for individual 1, party B for individual 2 and party C knows the selection
criteria. It is also acceptable if A knows both the relevant values & selection criteria. It can als be the case that
both A & B know relevant values due to a hybrid split. In this case missing records need to have a value of ?. E.g. A
has record 1,2,3 B has record 4,5,6 then the data needs to look as follows: A = [A1,A2,A3,?,?,?] B= [?,?,?,B1,B1,B1]
Similarly the attribute used for the selection criteria may be split up as well.

To use this method the following needs to be done:

1) setPrecision (default 5)
2) sumRelevantValues
3) postZ
4) sumZvalues

## Implemented methods:

### setPrecision:

Set the precision to be used for double values for the product protocol. Expected input:

- An int indicating the precision

Default precision used is 5. Always make sure to keep the precision the same across the various parties involved. To
keep everything alligned use setPrecisionCentral

### sumRelevantValues:

Sums the values based on the relevant individuals. Expected input:

- Predictor: the predicator to be summed
- Requirements: list of attributeRequirement indicating the relevant individuals


Example input:
```json
{
    "requirements": [{
        "value": {
            "type": "numeric",
            "value": "1",
            "attributeName": "x6",
            "id": null,
            "uknown": false
        },
        "range": false,
        "upperLimit": null,
        "lowerLimit": null,
        "name": "x6"
    }],
    "endpoint": "2"
}
```

### postZ:

Updates the local z-values. Expected input:

- z: array of doubles representing the z values
- endpoint: name of the relevant server
-

### sumZValues:

Sums the Z values based on the relevant individuals. Expected input:

- endpoint: the endpoint containing the relevant z values
- Requirements: list of attributeRequirement indicating the relevant individuals


### SOAPUI example project

A SOAPUI example project is also present in this repository.