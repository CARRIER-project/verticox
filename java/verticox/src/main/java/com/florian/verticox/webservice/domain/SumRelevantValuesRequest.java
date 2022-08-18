package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;

import java.util.List;

public class SumRelevantValuesRequest {
    private List<String> valueServers;
    private List<AttributeRequirement> requirements;

    public List<String> getValueServer() {
        return valueServers;
    }

    public void setValueServer(List<String> valueServers) {
        this.valueServers = valueServers;
    }

    public List<AttributeRequirement> getRequirements() {
        return requirements;
    }

    public void setRequirements(List<AttributeRequirement> requirements) {
        this.requirements = requirements;
    }
}
