package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.data.Data;
import com.florian.nscalarproduct.station.DataStation;
import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirementsRequest;
import com.florian.verticox.webservice.domain.InitCovariateRequest;
import com.florian.verticox.webservice.domain.MinimumPeriodRequest;
import com.florian.verticox.webservice.domain.UpdateCovariateRequest;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static com.florian.nscalarproduct.data.Parser.parseCsv;

public class VerticoxServer extends Server {
    private static final int DEFAULT_PRECISION = 5; //checkstyle's a bitch
    private static final int TEN = 10; //checkstyle's a bitch
    private int precision = DEFAULT_PRECISION; //precision for the n-party protocol since that works with integers
    private BigDecimal multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));

    private static final int MINIMUM_EVENT_POPULATION = 10;

    private Data data;
    private BigDecimal[] zValues;
    private String path;
    private BigDecimal[][] covariates;

    public VerticoxServer(String id) {
        this.serverId = id;
    }


    public VerticoxServer(String id, List<ServerEndpoint> endpoints) {
        this.serverId = id;
        this.setEndpoints(endpoints);
    }

    public VerticoxServer(String path, String id) {
        this.path = path;
        this.serverId = id;
        readData();
    }


    @PutMapping ("setZValues")
    public void setZValues(BigDecimal[] zValues) {
        this.zValues = zValues;
    }

    @PutMapping ("setPrecision")
    public void setPrecision(int precision) {
        this.precision = precision;
        multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
    }

    @PutMapping ("updateCovariateValues")
    public void updateCovariateValues(@RequestBody UpdateCovariateRequest req) {
        // This should require some calculation at some point, not quite sure where yet
        int column = this.data.getAttributeCollumn(req.getAttribute());
        for (int i = 0; i < population; i++) {
            this.covariates[column][i] = req.getCovariates()[i];
        }
    }

    @PutMapping ("initCovariateData")
    public void initCovariateData(@RequestBody InitCovariateRequest req) {
        reset();
        if (this.data == null) {
            readData();
        }
        localData = new BigInteger[population];
        int column = this.data.getAttributeCollumn(req.getAttribute());
        for (int i = 0; i < population; i++) {
            this.localData[i] = BigInteger.valueOf(this.covariates[column][i].multiply(multiplier).longValue());
        }
    }

    @PutMapping ("initZData")
    public void initZData() {
        reset();
        if (this.data == null) {
            readData();
        }
        localData = new BigInteger[population];
        for (int i = 0; i < population; i++) {
            localData[i] = BigInteger.valueOf(zValues[i].multiply(multiplier).longValue());
        }

        this.population = localData.length;
        this.dataStations.put("start", new DataStation(this.serverId, this.localData));
    }

    @GetMapping ("determineMinimumPeriod")
    public AttributeRequirement determineMinimumPeriod(@RequestBody MinimumPeriodRequest req) {
        //Assumption is that time T of events is represented by a real or integer value
        AttributeRequirement requirement = new AttributeRequirement();
        Attribute lower = req.getLowerLimit();
        requirement.setLowerLimit(lower);

        List<Attribute> unique = data.getAttributeValues(lower.getAttributeName());
        List<Attribute> sorted = unique.stream().sorted().collect(Collectors.toList());

        for (Attribute a : sorted) {
            // go through the sorted lists of values
            if (a.compareTo(lower) < 0) {
                //value is below minimum, ignore
                continue;
            } else {
                //value is above, or equal to minimum, attempt to use it as upperlimit
                requirement.setUpperLimit(a);
                if (countIndividuals(requirement) >= MINIMUM_EVENT_POPULATION) {
                    //found a range that contains sufficiently large population return requirement
                    return requirement;
                }
            }
        }
        //no sufficiently large population was found, return the maximum possible range
        return requirement;
    }

    private int countIndividuals(AttributeRequirement request) {
        int count = 0;
        for (Attribute value : data.getAttributeValues(request.getName())) {
            if (request.checkRequirement(value)) {
                count++;
            }
        }
        return count;
    }

    @PutMapping ("selectIndividuals")
    public void selectIndividuals(@RequestBody AttributeRequirementsRequest request) {
        //method to select appropriate individuals.
        //Assumption is that they're onl selected based on eventtime
        //But it is possible to select on multiple attributes at once
        reset();
        if (this.data == null) {
            readData();
        }
        localData = new BigInteger[population];
        for (int i = 0; i < population; i++) {
            localData[i] = BigInteger.ONE;
        }


        List<List<Attribute>> values = data.getData();
        for (AttributeRequirement req : request.getRequirements()) {
            if (data.getAttributeCollumn(req.getName()) == null) {
                // attribute not locally available, skip
                continue;
            }
            for (int i = 0; i < population; i++) {
                if (!req.checkRequirement(values.get(data.getAttributeCollumn(req.getName())).get(i))) {
                    localData[i] = BigInteger.ZERO;
                }
            }
        }

        this.population = localData.length;
        this.dataStations.put("start", new DataStation(this.serverId, this.localData));
    }

    private void readData() {
        if (System.getenv("DATABASE_URI") != null) {
            // Check if running in vantage6 by looking for system env, if yes change to database_uri system env for path
            this.path = System.getenv("DATABASE_URI");
        }
        this.data = parseCsv(path, 0);
        this.population = data.getNumberOfIndividuals();

        this.covariates = new BigDecimal[this.data.getData().size()][this.population];
    }

    @Override
    protected void reset() {
        dataStations = new HashMap<>();
        secretStations = new HashMap<>();
    }
}
