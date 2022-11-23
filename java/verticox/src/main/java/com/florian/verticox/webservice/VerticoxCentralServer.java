package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.station.CentralStation;
import com.florian.nscalarproduct.webservice.CentralServer;
import com.florian.nscalarproduct.webservice.Protocol;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.*;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
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

    private static final double BIN_UPPER_LIMIT_INCLUDE = 1.01;
    private static final int MINCOUNT = 10;

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

    @PostMapping ("postZ")
    public void postZ(@RequestBody InitZRequest z) {
        for (ServerEndpoint e : endpoints) {
            if (e.getServerId().equals(z.getEndpoint())) {
                ((VerticoxEndpoint) e).initZData(z.getZ());
            }
        }
    }

    @PostMapping ("getRelevantValues")
    public RelevantValuesResponse getRelevantValues(@RequestBody RelevantValueRequest req) {
        initEndpoints();
        Set<String> unique = new HashSet<>();
        Attribute.AttributeType type = null;
        for (ServerEndpoint endpoint : endpoints) {
            UniqueValueResponse response = ((VerticoxEndpoint) endpoint).getUniqueValues(req);
            unique.addAll(response.getUnique());
            if (response.getType() != null) {
                type = response.getType();
            }
        }
        RelevantValuesResponse response = new RelevantValuesResponse();
        response.setRelevantValues(createBins(unique, req.getAttribute(), type));
        return response;
    }

    private Set<Bin> createBins(Set<String> unique, String attribute, Attribute.AttributeType type) {
        Bin currentBin = new Bin();
        Bin lastBin = currentBin;
        Set<Bin> bins = new HashSet<>();
        if (unique.size() == 1) {
            //only one unique value
            String value = findSmallest(unique.stream().collect(Collectors.toList()), type);
            currentBin.setLower(value);
            currentBin.setUpper(value);
        } else {
            //set lowest lower limit
            String lower = findSmallest(unique.stream().collect(Collectors.toList()), type);
            removeValue(unique, lower);
            currentBin.setLower(lower);
            while (unique.size() > 0) {
                boolean lastBinToosmall = false;
                //create bins
                while (!binIsBigEnough(currentBin, attribute, type)) {
                    if (unique.size() == 0) {
                        //ran out of possible candidates for upperLimits before reaching a large enough bin
                        lastBinToosmall = true;
                        break;
                    }
                    //look for new upperlimit
                    String upper = findSmallest(unique.stream().collect(Collectors.toList()), type);
                    removeValue(unique, upper);
                    currentBin.setUpper(upper);
                }
                if (!lastBinToosmall) {
                    //found a upperLimit that makes the bin big enough
                    bins.add(currentBin);
                    lastBin = currentBin;
                    currentBin = new Bin();
                    //set lower limit to the previous upperLimit
                    currentBin.setLower(lastBin.getUpper());
                } else {
                    //did not find a large enough bin
                    //find last upper limit
                    String lastUpper = "";
                    if (currentBin.getUpper() != null) {
                        lastUpper = currentBin.getUpper();
                    } else {
                        lastUpper = lastBin.getUpper();
                    }
                    //increase the last upper Limit slightly so it is actually included
                    if (type == Attribute.AttributeType.numeric) {
                        //attribute is an integer
                        lastUpper = String.valueOf(Integer.parseInt(lastUpper) + 1);
                    } else {
                        //attribute is a double
                        lastUpper = String.valueOf(Double.parseDouble(lastUpper) * BIN_UPPER_LIMIT_INCLUDE);
                    }
                    lastBin.setUpper(lastUpper);
                    if (bins.size() == 0) {
                        //if the first bin is already too small, make sure to add it at this point
                        bins.add(lastBin);
                    }
                }

            }
        }
        return bins;
    }

    @PostMapping ("sumZValues")
    public double sumZValues(@RequestBody SumZRequest req) {
        initEndpoints();

        List<ServerEndpoint> relevantEndpoints = new ArrayList<>();
        BigDecimal divider = BigDecimal.ONE;
        for (ServerEndpoint endpoint : endpoints) {
            InitDataResponse response = ((VerticoxEndpoint) endpoint).initRt(req);
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
        Protocol prot = new Protocol(endpoints, secretEndpoint, "start", precision);
        return station.calculateNPartyScalarProduct(prot);
    }

    private String findSmallest(List<String> unique, Attribute.AttributeType type) {
        double smallest = Double.parseDouble(unique.get(0));
        for (int i = 1; i < unique.size(); i++) {
            double temp = Double.parseDouble(unique.get(i));
            if (temp < smallest) {
                smallest = temp;
            }
        }
        if (type == Attribute.AttributeType.numeric) {
            //manually turn into an int, otherwise java returns adds a .0
            return String.valueOf(((int) smallest));
        }
        return String.valueOf(smallest);
    }

    private boolean binIsBigEnough(Bin currentBin, String attribute, Attribute.AttributeType type) {
        if (currentBin.getUpper() == null || currentBin.getUpper().length() == 0) {
            //bin has no upper limit yet
            return false;
        } else {
            Attribute lower = new Attribute(type, currentBin.getLower(), attribute);
            Attribute upper = new Attribute(type, currentBin.getUpper(), attribute);

            AttributeRequirement req = new AttributeRequirement(lower, upper);

            SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
            request.setRequirements(Arrays.asList(req));
            List<ServerEndpoint> relevantEndpoints = new ArrayList<>();
            for (ServerEndpoint endpoint : endpoints) {
                InitDataResponse response = ((VerticoxEndpoint) endpoint).initData(request);
                if (response.isRelevant()) {
                    relevantEndpoints.add(endpoint);
                }
            }
            if (relevantEndpoints.size() == 1) {
                // only one relevant party, make things easy and just sum stuff
                return ((VerticoxEndpoint) relevantEndpoints.get(0)).getCount() >= MINCOUNT;
            } else {

                secretEndpoint.addSecretStation("start", relevantEndpoints.stream().map(x -> x.getServerId()).collect(
                        Collectors.toList()), relevantEndpoints.get(0).getPopulation());

                BigDecimal result = new BigDecimal(nparty(relevantEndpoints, secretEndpoint).toString());


                return result.intValue() >= MINCOUNT;
            }
        }
    }

    private void removeValue(Set<String> unique, String value) {
        //remove both the double and int variant of this value
        //doubles are automaticly translated into x.0 by java
        //if the original data simply contained x it'll get stuck in an endless loop
        unique.remove(value);
        unique.remove(String.valueOf((int) Double.parseDouble(value)));

    }
}
