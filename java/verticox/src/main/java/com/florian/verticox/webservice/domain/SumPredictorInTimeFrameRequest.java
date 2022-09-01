package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;

public class SumPredictorInTimeFrameRequest {
    private AttributeRequirement timeFrame;
    private String predictor;

    public AttributeRequirement getTimeFrame() {
        return timeFrame;
    }

    public void setTimeFrame(AttributeRequirement timeFrame) {
        this.timeFrame = timeFrame;
    }

    public String getPredictor() {
        return predictor;
    }

    public void setPredictor(String predictor) {
        this.predictor = predictor;
    }
}
