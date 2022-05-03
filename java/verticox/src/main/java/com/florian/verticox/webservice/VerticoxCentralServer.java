package com.florian.verticox.webservice;

import com.florian.nscalarproduct.station.CentralStation;
import com.florian.nscalarproduct.webservice.CentralServer;
import com.florian.nscalarproduct.webservice.Protocol;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.InitCentralServerRequest;
import com.florian.verticox.webservice.domain.MinimumPeriodRequest;
import com.florian.verticox.webservice.domain.SumRelevantValuesRequest;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
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

    private void initEndpoints() {
        if (endpoints.size() == 0) {
            endpoints = new ArrayList<>();
            for (String s : servers) {
                endpoints.add(new VerticoxEndpoint(s));
            }
        }
        if (secretEndpoint == null) {
            secretEndpoint = new ServerEndpoint(secretServer);
        }
    }

    @PostMapping ("initCentralServer")
    public void initCentralServer(@RequestBody InitCentralServerRequest req) {
        //purely exists for vantage6
        super.secretServer = req.getSecretServer();
        super.servers = req.getServers();
    }

    @PutMapping ("setPrecisionCentral")
    public void setPrecisionCentral(int precision) {
        initEndpoints();
        this.precision = precision;
        multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
        endpoints.stream().forEach(x -> ((VerticoxEndpoint) x).setPrecision(precision));
    }

    @PostMapping ("determineMinimumPeriodCentral")
    public AttributeRequirement determineMinimumPeriodCentral(@RequestBody MinimumPeriodRequest req) {
        initEndpoints();
        List<AttributeRequirement> list = endpoints.stream()
                .map(x -> ((VerticoxEndpoint) x).determineMinimumPeriod(req.getLowerLimit())).collect(
                        Collectors.toList()).stream().filter(Objects::nonNull).collect(Collectors.toList());
        //If the attribute exists it only exists in one place, so return first value in the list, the rest was null
        //if it doesn't exist return null
        if (list.size() > 0) {
            return list.get(0);
        }
        return null;
    }

    @GetMapping ("sumRelevantValues")
    public BigDecimal sumRelevantValues(@RequestBody SumRelevantValuesRequest req) {
        initEndpoints();
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
