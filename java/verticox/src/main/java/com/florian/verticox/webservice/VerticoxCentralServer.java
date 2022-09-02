package com.florian.verticox.webservice;

import com.florian.nscalarproduct.station.CentralStation;
import com.florian.nscalarproduct.webservice.CentralServer;
import com.florian.nscalarproduct.webservice.Protocol;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.verticox.webservice.domain.InitCentralServerRequest;
import com.florian.verticox.webservice.domain.InitDataResponse;
import com.florian.verticox.webservice.domain.SumPredictorInTimeFrameRequest;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
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
        endpoints.stream().forEach(x -> x.initEndpoints());
        secretEndpoint.initEndpoints();
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

    @PostMapping ("sumRelevantValues")
    public double sumRelevantValues(@RequestBody SumPredictorInTimeFrameRequest req) {
        initEndpoints();

        List<ServerEndpoint> relevantEndpoints = new ArrayList<>();
        BigDecimal divider = BigDecimal.ONE;
        for (ServerEndpoint endpoint : endpoints) {
            InitDataResponse response = ((VerticoxEndpoint) endpoint).initData(req);
            if (response.isRelevant()) {
                relevantEndpoints.add(endpoint);
                if (response.isPredictorPresent()) {
                    divider = divider.multiply(multiplier);
                }
            }
        }

        if (relevantEndpoints.size() == 1) {
            // only one relevant party, make things easy and just sum stuff
            return ((VerticoxEndpoint) relevantEndpoints.get(0)).getSum();
        } else {

            secretEndpoint.addSecretStation("start", relevantEndpoints.stream().map(x -> x.getServerId()).collect(
                    Collectors.toList()), relevantEndpoints.get(0).getPopulation());

            BigDecimal result = new BigDecimal(nparty(relevantEndpoints, secretEndpoint).toString());


            return result.divide(divider).doubleValue();
        }
    }

    private BigInteger nparty(List<ServerEndpoint> endpoints, ServerEndpoint secretEndpoint) {
        CentralStation station = new CentralStation();
        Protocol prot = new Protocol(endpoints, secretEndpoint, "start");
        return station.calculateNPartyScalarProduct(prot);
    }
}
