package com.florian.verticox.webservice.domain;

import java.util.HashSet;
import java.util.Set;

public class RelevantValuesResponse {
    private Set<Bin> relevantValues = new HashSet<>();

    public Set<Bin> getRelevantValues() {
        return relevantValues;
    }

    public void setRelevantValues(Set<Bin> relevantValues) {
        this.relevantValues = relevantValues;
    }
}
