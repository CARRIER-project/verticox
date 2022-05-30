package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;

import java.util.List;

public class SumRelevantValuesRequest {
    private String valueServer;
    private List<AttributeRequirement> requirements;

    public String getValueServer() {
        return valueServer;
    }

    public void setValueServer(String valueServer) {
        this.valueServer = valueServer;
    }

    public List<AttributeRequirement> getRequirements() {
        return requirements;
    }

    public void setRequirements(List<AttributeRequirement> requirements) {
        this.requirements = requirements;
    }
}
