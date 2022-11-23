package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.data.Attribute;

import java.util.HashSet;
import java.util.Set;

public class UniqueValueResponse {
    private Set<String> unique = new HashSet<>();
    private Attribute.AttributeType type;

    public Attribute.AttributeType getType() {
        return type;
    }

    public void setType(Attribute.AttributeType type) {
        this.type = type;
    }

    public Set<String> getUnique() {
        return unique;
    }

    public void setUnique(Set<String> unique) {
        this.unique = unique;
    }
}
