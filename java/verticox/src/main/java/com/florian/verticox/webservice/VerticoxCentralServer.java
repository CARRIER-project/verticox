package com.florian.verticox.webservice;

import com.florian.nscalarproduct.station.CentralStation;
import com.florian.nscalarproduct.webservice.CentralServer;
import com.florian.nscalarproduct.webservice.Protocol;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import org.springframework.web.bind.annotation.PutMapping;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
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

    @PutMapping ("setPrecision")
    public void setPrecision(int precision) {
        this.precision = precision;
        multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
    }

    public BigDecimal sumRelevantZ(String zServer, AttributeRequirement relevantT) {
        for (ServerEndpoint endpoint : endpoints) {
            if (endpoint.getServerId().equals(zServer)) {
                ((VerticoxEndpoint) endpoint).initZData();
            } else {
                ((VerticoxEndpoint) endpoint).selectIndividuals(Arrays.asList(relevantT));
            }
        }
        secretEndpoint.addSecretStation("start", endpoints.stream().map(x -> x.getServerId()).collect(
                Collectors.toList()), endpoints.get(0).getPopulation());


        return BigDecimal.valueOf(nparty(endpoints, secretEndpoint).longValue())
                .divide(multiplier);
    }

    public BigInteger nparty(List<ServerEndpoint> endpoints, ServerEndpoint secretEndpoint) {
        CentralStation station = new CentralStation();
        Protocol prot = new Protocol(endpoints, secretEndpoint, "start");
        return station.calculateNPartyScalarProduct(prot);
    }
}
