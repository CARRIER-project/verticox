package com.florian.verticox.webservice.domain;

import java.util.List;

public class InitCentralServerRequest {
    private String secretServer;
    private List<String> servers;

    public InitCentralServerRequest() {
    }

    public String getSecretServer() {
        return secretServer;
    }

    public void setSecretServer(String secretServer) {
        this.secretServer = secretServer;
    }

    public List<String> getServers() {
        return servers;
    }

    public void setServers(List<String> servers) {
        this.servers = servers;
    }
}
