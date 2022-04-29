package com.florian.verticox.webservice;

import com.florian.nscalarproduct.station.CentralStation;
import com.florian.nscalarproduct.webservice.CentralServer;
import com.florian.nscalarproduct.webservice.Protocol;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.verticox.webservice.domain.InitCentralServerRequest;
import com.florian.verticox.webservice.domain.SumRelevantValuesRequest;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class VerticoxCentralServer extends CentralServer {
    private List<ServerEndpoint> endpoints = new ArrayList<>();
    private ServerEndpoint secretEndpoint;
    private boolean testing;

    private static final int DEFAULT_PRECISION = 5; //checkstyle's a bitch
    private static final int TEN = 10; //checkstyle's a bitch
    private int precision = DEFAULT_PRECISION; //precision for the n-party protocol since that works with integers
    private BigDecimal multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));

    public VerticoxCentralServer() {
    }

    public VerticoxCentralServer(boolean testing) {
        this.testing = testing;
    }

    public void initEndpoints(List<ServerEndpoint> endpoints, ServerEndpoint secretServer) {
        //this only exists for testing purposes
        this.endpoints = endpoints;
        this.secretEndpoint = secretServer;
    }

    @PostMapping ("initCentralServer")
    public void initCentralServer(@RequestBody InitCentralServerRequest req) {
        //purely exists for vantage6
        super.secretServer = req.getSecretServer();
        super.servers = req.getServers();
    }

    @PutMapping ("setPrecision")
    public void setPrecision(int precision) {
        this.precision = precision;
        multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
    }

    @GetMapping ("sumRelevantValues")
    public BigDecimal sumRelevantValues(@RequestBody SumRelevantValuesRequest req) {
        for (ServerEndpoint endpoint : endpoints) {
            if (endpoint.getServerId().equals(req.getValueServer())) {
                ((VerticoxEndpoint) endpoint).initData(req.getRequirements());
            } else {
                ((VerticoxEndpoint) endpoint).selectIndividuals(req.getRequirements());
            }
        }
        secretEndpoint.addSecretStation("start", endpoints.stream().map(x -> x.getServerId()).collect(
                Collectors.toList()), endpoints.get(0).getPopulation());


        return BigDecimal.valueOf(nparty(endpoints, secretEndpoint).longValue())
                .divide(multiplier);
    }

    private BigInteger nparty(List<ServerEndpoint> endpoints, ServerEndpoint secretEndpoint) {
        CentralStation station = new CentralStation();
        Protocol prot = new Protocol(endpoints, secretEndpoint, "start");
        return station.calculateNPartyScalarProduct(prot);
    }
}
