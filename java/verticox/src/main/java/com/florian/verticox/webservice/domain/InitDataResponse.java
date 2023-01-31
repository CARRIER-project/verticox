package com.florian.verticox.webservice.domain;

public class InitDataResponse {
    private boolean predictorPresent;
    private boolean outcomePresent;

    public boolean isPredictorPresent() {
        return predictorPresent;
    }

    public void setPredictorPresent(boolean predictorPresent) {
        this.predictorPresent = predictorPresent;
    }

    public boolean isOutcomePresent() {
        return outcomePresent;
    }

    public void setOutcomePresent(boolean outcomePresent) {
        this.outcomePresent = outcomePresent;
    }

    public boolean isRelevant() {
        return outcomePresent || predictorPresent;
    }
}
