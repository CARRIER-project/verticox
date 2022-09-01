package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.SumPredictorInTimeFrameRequest;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VerticoxEndpointTest {

    private Cipher cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");

    public VerticoxEndpointTest() throws NoSuchPaddingException, NoSuchAlgorithmException {
    }

    @Test
    public void testValuesMultiplicationX2_PredictorAndOutComeLocal()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x2");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX3_PredictorAndOutcomeSplit()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x3");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX3_Range_LargerThan1()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setRange(true);
        req.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        req.setUpperLimit(new Attribute(Attribute.AttributeType.numeric, "inf", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x3");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX3_Range_SmallerThan1()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);


        Attribute lower = new Attribute(Attribute.AttributeType.numeric, "-2", "x1");
        Attribute upper = new Attribute(Attribute.AttributeType.numeric, "1", "x1");
        AttributeRequirement req = new AttributeRequirement(lower, upper);
        //this selects individuals: 3,5,6,8, & 10

        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x3");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 2;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX3_Range_Everything()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);


        Attribute lower = new Attribute(Attribute.AttributeType.numeric, "-inf", "x1");
        Attribute upper = new Attribute(Attribute.AttributeType.numeric, "10", "x1");
        AttributeRequirement req = new AttributeRequirement(lower, upper);
        //this selects all individuals

        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x3");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 6;

        assertEquals(result.longValue(), expected);
    }


    @Test
    public void testValuesMultiplicationX3_PredictorAndOutcomeSplit_ThreeServers()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer("resources/smallK2Example_thirdhalf.csv", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("secret", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        all.add(endpoint3);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);
        server3.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x3");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX4_PredictorHybridSplit()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x4");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 10;

        assertEquals(result.longValue(), expected);
    }

    @Test
    public void testValuesMultiplicationX5_PredictorHybridSplit_ThreeServers()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer("resources/smallK2Example_thirdhalf.csv", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9


        VerticoxServer secret = new VerticoxServer("secret", Arrays.asList(endpointZ, endpoint2, endpoint3));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointZ);
        all.add(endpoint2);
        all.add(endpoint3);
        all.add(secretEnd);
        secret.setEndpoints(all);
        serverZ.setEndpoints(all);
        server2.setEndpoints(all);
        server3.setEndpoints(all);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        int precision = 5;
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setTimeFrame(req);
        request.setPredictor("x5");
        BigDecimal result = central.sumRelevantValues(request);
        int expected = 10;

        assertEquals(result.longValue(), expected);
    }
}