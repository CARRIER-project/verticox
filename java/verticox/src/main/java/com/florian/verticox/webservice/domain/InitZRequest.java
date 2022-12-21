package com.florian.verticox.webservice.domain;

public class InitZRequest {
    private String attribute;

    public String getAttribute() {
        return attribute;
    }

    public void setAttribute(String attribute) {
        this.attribute = attribute;
    }

    private double[] z;

    public double[] getZ() {
        return z;
    }

    public void setZ(double[] z) {
        this.z = z;
    }
}
