package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.*;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;
import java.io.UnsupportedEncodingException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x2");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplicationX2_PredictorAndOutComeLocal_TestFold()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9

        ActiveRecordRequest active = new ActiveRecordRequest();
        // only activate the odd ones, resuling in 1,7 & 9
        active.setActiveRecords(
                new boolean[] {true, false, true, false, true, false, true, false, true, false});


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
        central.activateFold(active);

        int precision = 5;

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x2");
        double result = central.sumRelevantValues(request);
        int expected = 2;

        assertEquals(result, expected);

        // only activate the even ones, resuling in 2,4
        active.setActiveRecords(
                new boolean[] {false, true, false, true, false, true, false, true, false, true});
        central.activateFold(active);
        request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x2");
        result = central.sumRelevantValues(request);
        expected = 2;

        assertEquals(result, expected);
        // only activate the the first 7, resulting in 1,2,4,7
        active.setActiveRecords(
                new boolean[] {true, true, true, true, true, true, true, false, false, false});
        central.activateFold(active);
        request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x2");
        result = central.sumRelevantValues(request);
        expected = 3;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplicationX2_RequirementRange()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);


        Attribute upper = new Attribute(Attribute.AttributeType.numeric, "10", "x1");
        Attribute lower = new Attribute(Attribute.AttributeType.numeric, "1", "x1");
        AttributeRequirement req = new AttributeRequirement(lower, upper);
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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x2");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
    }

    @Test
    public void testRelevantValues()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/bigK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/bigK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);
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

        RelevantValueRequest request = new RelevantValueRequest();
        request.setAttribute("x4");

        RelevantValuesResponse result = central.getRelevantValues(request);
        assertEquals(result.getRelevantValues().size(), 3);

        // 3 bins: 0-1, 1-2, 2-18 check these exist
        boolean found_01 = false;
        boolean found_12 = false;
        boolean found_218 = false;
        for (Bin bin : result.getRelevantValues()) {
            if (bin.getLower().equals("0")) {
                assertEquals(bin.getUpper(), "1");
                found_01 = true;
            } else if (bin.getLower().equals("1")) {
                assertEquals(bin.getUpper(), "2");
                found_12 = true;
            } else if (bin.getLower().equals("2")) {
                assertEquals(bin.getUpper(), "18");
                found_218 = true;
            }
        }

        assertTrue(found_01);
        assertTrue(found_12);
        assertTrue(found_218);
    }

    @Test
    public void testRelevantValuesHybridSplit()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/bigK2Example_secondhalf_hybrid_1.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/bigK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer("resources/bigK2Example_secondhalf_hybrid_2.csv", "4");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);
        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2, endpoint3));
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        RelevantValueRequest request = new RelevantValueRequest();
        request.setAttribute("x4");
        RelevantValuesResponse result = central.getRelevantValues(request);
        assertEquals(result.getRelevantValues().size(), 3);

        // 3 bins: 0-1, 1-2, 2-18 check these exist
        boolean found_01 = false;
        boolean found_12 = false;
        boolean found_218 = false;
        for (Bin bin : result.getRelevantValues()) {
            if (bin.getLower().equals("0")) {
                assertEquals(bin.getUpper(), "1");
                found_01 = true;
            } else if (bin.getLower().equals("1")) {
                assertEquals(bin.getUpper(), "2");
                found_12 = true;
            } else if (bin.getLower().equals("2")) {
                assertEquals(bin.getUpper(), "18");
                found_218 = true;
            }
        }

        assertTrue(found_01);
        assertTrue(found_12);
        assertTrue(found_218);
    }

    @Test
    public void testValuesMultiplicationX3_MultipleRequirements()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        AttributeRequirement req2 = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        req2.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x2"));
        //this selects individuals: 2,4,7, & 9


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

        int precision = 4;

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req, req2));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplicationX3_SplitRequirements()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        AttributeRequirement req2 = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        req2.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x3"));
        //this selects individuals: 4,7, & 9


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

        int precision = 4;

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req, req2));
        request.setPredictor("x4");
        double result = central.sumRelevantValues(request);
        int expected = 8;

        assertEquals(result, expected);
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 2;

        assertEquals(result, expected);
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 6;

        assertEquals(result, expected);
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

        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2, endpoint3), secretEnd);

        SumPredictorInTimeFrameRequest request = new SumPredictorInTimeFrameRequest();
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x3");
        double result = central.sumRelevantValues(request);
        int expected = 4;

        assertEquals(result, expected);
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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x4");
        double result = central.sumRelevantValues(request);
        int expected = 10;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplicationX7_Decimals()
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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x7");
        double result = central.sumRelevantValues(request);
        double expected = 1.818;

        assertEquals(result, expected);
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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x5");
        double result = central.sumRelevantValues(request);
        int expected = 10;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplicationX6_SelectorHybridSplit_ThreeServers()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer("resources/smallK2Example_thirdhalf.csv", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x6"));
        //this selects individuals: 2,4,6,8, & 10


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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("x5");
        double result = central.sumRelevantValues(request);
        int expected = 10;

        assertEquals(result, expected);
    }

    @Test
    public void testAllFileTypes()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/mixedFiles/allTypes.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/mixedFiles/allTypes.arff", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer("resources/mixedFiles/allTypes.parquet", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.real, "1", "real_arff"));
        //selects record 1,2,4,7,9

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
        request.setRequirements(Arrays.asList(req));
        request.setPredictor("numeric_parquet");
        double result = central.sumRelevantValues(request);
        int expected = 23;

        assertEquals(result, expected);
    }

    @Test
    public void testinitCentralServerRequest() {

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        InitCentralServerRequest req = new InitCentralServerRequest();
        req.setSecretServer("secret");
        req.setServers(Arrays.asList("1", "2"));
        central.initCentralServer(req);
    }
}