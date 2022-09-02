package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;

import java.util.List;

public class SumPredictorInTimeFrameRequest {
    private List<AttributeRequirement> requirements;
    private String predictor;

    public List<AttributeRequirement> getRequirements() {
        return requirements;
    }

    public void setRequirements(List<AttributeRequirement> requirements) {
        this.requirements = requirements;
    }

    public String getPredictor() {
        return predictor;
    }

    public void setPredictor(String predictor) {
        this.predictor = predictor;
    }
}
