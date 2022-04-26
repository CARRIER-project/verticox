package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VerticoxEndpointTest {

    @Test
    public void testZValuesMultiplication() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] zValues = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            zValues[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setZValues(zValues);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9
        BigDecimal expected = BigDecimal.ZERO;
        expected = expected.add(zValues[0]).add(zValues[1]).add(zValues[3]).add(zValues[6]).add(zValues[8]);

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

        BigDecimal result = central.sumRelevantZ("Z", req);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testZValuesMultiplicationInfiniteRange() {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] zValues = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            zValues[i] = BigDecimal.valueOf(r.nextDouble());
        }
        endpointZ.setZValues(zValues);

        AttributeRequirement req = new AttributeRequirement(new Attribute(Attribute.AttributeType.numeric, "0", "x1"),
                                                            new Attribute(Attribute.AttributeType.numeric, "inf",
                                                                          "x1"));
        //this selects all individuals
        BigDecimal expected = Arrays.stream(zValues).reduce(BigDecimal.ZERO, BigDecimal::add);

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

        BigDecimal result = central.sumRelevantZ("Z", req);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

}