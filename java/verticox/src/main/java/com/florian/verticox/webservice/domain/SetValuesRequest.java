package com.florian.verticox.webservice.domain;

public class SetValuesRequest {
    private String[] values;
    private byte[] encryptedAes;

    public String[] getValues() {
        return values;
    }

    public void setValues(String[] values) {
        this.values = values;
    }

    public byte[] getEncryptedAes() {
        return encryptedAes;
    }

    public void setEncryptedAes(byte[] encryptedAes) {
        this.encryptedAes = encryptedAes;
    }
}
