package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.SumRelevantValuesRequest;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VerticoxEndpointTest {

    @Test
    public void testValuesMultiplication() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setValues(values);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9
        BigDecimal expected = BigDecimal.ZERO;
        expected = expected.add(values[0]).add(values[1]).add(values[3]).add(values[6]).add(values[8]);

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

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecision(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationHybridSplit() {
        VerticoxServer serverZ = new VerticoxServer("resources/hybridsplit/smallK2Example_firsthalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/hybridsplit/smallK2Example_secondhalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer(
                "resources/hybridsplit/smallK2Example_secondhalf_morepopulation.csv", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setValues(values);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x2"));
        //this selects individuals: 2,3,4,6,7, & 9
        BigDecimal expected = BigDecimal.ZERO;
        expected = expected.add(values[1]).add(values[2]).add(values[3]).add(values[5]).add(values[7]).add(values[8]);

        VerticoxServer secret = new VerticoxServer("4", Arrays.asList(endpointZ, endpoint2, endpoint3));
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

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecision(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationThreeServers() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);
        VerticoxServer server3 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setValues(values);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9
        BigDecimal expected = BigDecimal.ZERO;
        expected = expected.add(values[0]).add(values[1]).add(values[3]).add(values[6]).add(values[8]);

        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointZ, endpoint2, endpoint3));
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

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        endpoint3.setPrecision(precision);
        central.setPrecision(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationInfiniteRange() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setValues(values);

        AttributeRequirement req = new AttributeRequirement(new Attribute(Attribute.AttributeType.numeric, "0", "x1"),
                                                            new Attribute(Attribute.AttributeType.numeric, "inf",
                                                                          "x1"));
        //this selects all individuals
        BigDecimal expected = Arrays.stream(values).reduce(BigDecimal.ZERO, BigDecimal::add);

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

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecision(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationDataAndCriteriaInSamePlace() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpoint2.setValues(values);

        AttributeRequirement req = new AttributeRequirement(new Attribute(Attribute.AttributeType.numeric, "0", "x1"),
                                                            new Attribute(Attribute.AttributeType.numeric, "inf",
                                                                          "x1"));
        //this selects all individuals
        BigDecimal expected = Arrays.stream(values).reduce(BigDecimal.ZERO, BigDecimal::add);

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

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecision(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

}