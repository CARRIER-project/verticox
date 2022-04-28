package com.florian.verticox.webservice.domain;

import java.math.BigDecimal;

public class UpdateCovariateRequest {
    private String attribute;
    private BigDecimal[] covariates;

    public String getAttribute() {
        return attribute;
    }

    public void setAttribute(String attribute) {
        this.attribute = attribute;
    }

    public BigDecimal[] getCovariates() {
        return covariates;
    }

    public void setCovariates(BigDecimal[] covariates) {
        this.covariates = covariates;
    }
}
