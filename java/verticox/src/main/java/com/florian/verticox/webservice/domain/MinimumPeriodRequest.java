package com.florian.verticox.webservice.domain;

import com.florian.nscalarproduct.data.Attribute;

public class MinimumPeriodRequest {
    private Attribute lowerLimit;

    public Attribute getLowerLimit() {
        return lowerLimit;
    }

    public void setLowerLimit(Attribute lowerLimit) {
        this.lowerLimit = lowerLimit;
    }
}
