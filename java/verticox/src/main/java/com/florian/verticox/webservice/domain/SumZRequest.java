package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;

import java.util.List;

public class SumZRequest {
    private String endpoint;
    private List<AttributeRequirement> requirements;

    public String getEndpoint() {
        return endpoint;
    }

    public void setEndpoint(String endpoint) {
        this.endpoint = endpoint;
    }

    public List<AttributeRequirement> getRequirements() {
        return requirements;
    }

    public void setRequirements(List<AttributeRequirement> requirements) {
        this.requirements = requirements;
    }
}
