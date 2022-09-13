package com.florian.verticox.webservice.domain;

public class InitZRequest {
    private String endpoint;
    private double[] z;

    public double[] getZ() {
        return z;
    }

    public void setZ(double[] z) {
        this.z = z;
    }

    public String getEndpoint() {
        return endpoint;
    }

    public void setEndpoint(String endpoint) {
        this.endpoint = endpoint;
    }
}
