package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.data.Data;
import com.florian.nscalarproduct.encryption.RSA;
import com.florian.nscalarproduct.error.InvalidDataFormatException;
import com.florian.nscalarproduct.station.DataStation;
import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;

import javax.crypto.NoSuchPaddingException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.List;

import static com.florian.nscalarproduct.data.Parser.parseData;


@RestController
public class VerticoxServer extends Server {
    private static final int DEFAULT_PRECISION = 5; //checkstyle's a bitch
    private static final int TEN = 10; //checkstyle's a bitch
    private int precision = DEFAULT_PRECISION; //precision for the n-party protocol since that works with integers
    private BigDecimal multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
    private final RSA rsa = new RSA();
    private BigInteger[] z;

    private static final int MINIMUM_EVENT_POPULATION = 10;

    private Data data;
    private BigDecimal[] values;

    @Value ("${datapath}")
    private String path;

    public VerticoxServer() throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        super();
    }

    public VerticoxServer(String id)
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        this.serverId = id;
    }


    public VerticoxServer(String id, List<ServerEndpoint> endpoints)
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        this.serverId = id;
        this.setEndpoints(endpoints);
    }

    public VerticoxServer(String path, String id)
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        this.path = path;
        this.serverId = id;
        readData();
    }

    @PutMapping ("setPrecision")
    public void setPrecision(int precision) {
        this.precision = precision;
        multiplier = BigDecimal.valueOf(Math.pow(TEN, precision));
    }

    @GetMapping ("getSum")
    public double getSum() {
        BigInteger sum = BigInteger.ZERO;
        for (BigInteger d : this.localData) {
            sum = sum.add(d);
        }
        return new BigDecimal(sum).divide(multiplier).doubleValue();
    }

    @GetMapping ("getCount")
    public Integer getCount() {
        BigInteger sum = BigInteger.ZERO;
        for (BigInteger d : this.localData) {
            sum = sum.add(d);
        }
        return sum.intValue();
    }

    @PostMapping ("getUniqueValues")
    public UniqueValueResponse getUniqueValues(@RequestBody RelevantValueRequest req) {
        UniqueValueResponse res = new UniqueValueResponse();
        if (this.data == null) {
            readData();
        }
        if (data.getAttributeCollumn(req.getAttribute()) != null) {
            res.setUnique(data.getUniqueValues(data.getAttributeValues(req.getAttribute())));
            res.setType(data.getAttributeType(req.getAttribute()));
        }
        return res;
    }

    @PostMapping ("initData")
    public InitDataResponse initData(@RequestBody SumPredictorInTimeFrameRequest request) {
        reset();
        if (this.data == null) {
            readData();
        }

        boolean predictorPresent = isLocallyPresent(request.getPredictor());
        boolean requirementPresent = isRequirementPresent(request.getRequirements());

        if (requirementPresent) {
            // If the time variable is locally present select individuals
            selectIndividuals(request.getRequirements());
        }
        if (predictorPresent) {
            // If predictor is locally present
            List<Attribute> values = this.data.getAttributeValues(request.getPredictor());
            if (requirementPresent) {
                // Time variable was present, so multiply all selected population with the predictor value
                for (int i = 0; i < population; i++) {
                    // selected population currently has localData = 1
                    // Not selected currently has localData = 0
                    // This way if the criteria & data are in the same location only the applicable population is
                    // selected
                    localData[i] = localData[i].multiply(transForm(values.get(i)));
                }
            } else {
                // Time variable was not present, so just insert the predictor value
                localData = new BigInteger[population];
                for (int i = 0; i < population; i++) {
                    localData[i] = transForm(values.get(i));
                }
            }
        }
        if (predictorPresent || requirementPresent) {
            this.population = localData.length;
            this.dataStations.put("start", new DataStation(this.serverId, this.localData));
        }
        InitDataResponse response = new InitDataResponse();
        response.setOutcomePresent(requirementPresent);
        response.setPredictorPresent(predictorPresent);
        return response;
    }

    private boolean isRequirementPresent(
            List<AttributeRequirement> requirements) {
        boolean requirementPresent = false;
        for (AttributeRequirement r : requirements) {
            if (isLocallyPresent(r.getName())) {
                // there can theoretically be multiple requirements, only care if at least 1 is locally present
                requirementPresent = true;
                break;
            }
        }
        return requirementPresent;
    }

    @PostMapping ("initRt")
    public InitDataResponse initRt(@RequestBody SumZRequest request) {
        reset();
        if (this.data == null) {
            readData();
        }
        boolean zLocal = false;
        if (request.getEndpoint().equals(this.getServerId())) {
            zLocal = true;
        }

        boolean requirementPresent = isRequirementPresent(request.getRequirements());

        if (requirementPresent) {
            // If the time variable is locally present select individuals
            selectIndividuals(request.getRequirements());
        }
        if (zLocal) {
            // If z is locally present
            if (requirementPresent) {
                // Time variable was present, so multiply all selected population with the predictor value
                for (int i = 0; i < population; i++) {
                    // selected population currently has localData = 1
                    // Not selected currently has localData = 0
                    // This way if the criteria & data are in the same location only the applicable population is
                    // selected
                    localData[i] = localData[i].multiply(z[i]);
                }
            } else {
                // Time variable was not present, so just insert the predictor value
                localData = new BigInteger[population];
                for (int i = 0; i < population; i++) {
                    localData[i] = z[i];
                }
            }
        }
        if (zLocal || requirementPresent) {
            this.population = localData.length;
            this.dataStations.put("start", new DataStation(this.serverId, this.localData));
        }

        InitDataResponse response = new InitDataResponse();
        response.setOutcomePresent(requirementPresent);
        response.setPredictorPresent(zLocal);
        return response;
    }

    @PutMapping ("initZData")
    public void initZData(@RequestBody double[] z) {
        reset();
        this.z = new BigInteger[z.length];
        for (int i = 0; i < z.length; i++) {
            this.z[i] = new BigDecimal(String.valueOf(z[i])).multiply(multiplier).toBigIntegerExact();
        }
    }

    private BigInteger transForm(Attribute attribute) {
        if (attribute.isUnknown()) {
            //if locally unknown set to 1, another party will set it to the correct value and 1xvalue=value
            return BigDecimal.ONE.multiply(multiplier).toBigIntegerExact();
        } else {
            //if locally known set the value to whatever the value is multiplied with the multiplier for precision
            return new BigDecimal(attribute.getValue()).multiply(multiplier).toBigIntegerExact();
        }
    }

    private boolean isLocallyPresent(String predictor) {
        return this.data.getAttributeCollumn(predictor) != null;
    }

    private void selectIndividuals(List<AttributeRequirement> reqs) {
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
        for (AttributeRequirement req : reqs) {
            for (int i = 0; i < population; i++) {
                if (isLocallyPresent(req.getName())) {
                    Attribute a = values.get(data.getAttributeCollumn(req.getName())).get(i);
                    if (locallyUnknown(a)) {
                        // attribute is locally unknown so ignore it in this vector, another party will correct this
                        continue;
                    } else if (!req.checkRequirement(a)) {
                        // attribute is locally known and the check fails
                        localData[i] = BigInteger.ZERO;
                    }
                }
            }
        }

        checkHorizontalSplit(data, localData);

        this.population = localData.length;
        this.dataStations.put("start", new DataStation(this.serverId, this.localData));
    }

    private boolean locallyUnknown(Attribute a) {
        return a.isUnknown();
    }

    private void readData() {
        if (System.getenv("DATABASE_URI") != null) {
            // Check if running in vantage6 by looking for system env, if yes change to database_uri system env for path
            this.path = System.getenv("DATABASE_URI");
        }
        try {
            this.data = parseData(path, 0);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvalidDataFormatException e) {
            e.printStackTrace();
        }
        this.population = data.getNumberOfIndividuals();
    }

    @Override
    protected void reset() {
        dataStations = new HashMap<>();
        secretStations = new HashMap<>();
    }
}
