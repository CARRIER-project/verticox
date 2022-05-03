package com.florian.verticox.webservice;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.encryption.AES;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.MinimumPeriodRequest;
import com.florian.verticox.webservice.domain.SetValuesRequest;
import com.florian.verticox.webservice.domain.SumRelevantValuesRequest;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class VerticoxEndpointTest {

    private Cipher cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");

    public VerticoxEndpointTest() throws NoSuchPaddingException, NoSuchAlgorithmException {
    }

    @Test
    public void testValuesMultiplication()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = generateValues();

        endpointZ.setValues(createSetValuesRequest(endpointZ, values));

        ObjectMapper mapper = new ObjectMapper();
        try {
            System.out.println(mapper.writerWithDefaultPrettyPrinter()
                                       .writeValueAsString(createSetValuesRequest(endpointZ, values)));
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }

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
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    private BigDecimal[] generateValues() {
        BigDecimal[] values = new BigDecimal[10];
        Random r = new Random();
        for (int i = 0; i < 10; i++) {
            values[i] = BigDecimal.valueOf(r.nextDouble());
        }
        return values;
    }

    @Test
    public void testMinimumPeriod()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer smalExample = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "Z");
        VerticoxEndpoint endpointsmall = new VerticoxEndpoint(smalExample);

        VerticoxServer bigExample = new VerticoxServer("resources/bigK2Example_firsthalf.csv", "Z");
        VerticoxServer bigExampleSecondHalf = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z2");

        VerticoxEndpoint endpointBig = new VerticoxEndpoint(bigExample);
        VerticoxEndpoint endpointSecondHalf = new VerticoxEndpoint(bigExampleSecondHalf);

        VerticoxServer secret = new VerticoxServer("3", Arrays.asList(endpointBig, endpointSecondHalf));
        ServerEndpoint secretEnd = new ServerEndpoint(secret);

        List<ServerEndpoint> all = new ArrayList<>();
        all.add(endpointBig);
        all.add(endpointSecondHalf);

        VerticoxCentralServer central = new VerticoxCentralServer(true);
        central.initEndpoints(Arrays.asList(endpointBig, endpointSecondHalf), secretEnd);


        AttributeRequirement resultSmall = endpointsmall.determineMinimumPeriod(
                new Attribute(Attribute.AttributeType.numeric, "0", "x1"));

        MinimumPeriodRequest req1 = new MinimumPeriodRequest();
        req1.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "0", "x1"));
        MinimumPeriodRequest req2 = new MinimumPeriodRequest();
        req2.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        MinimumPeriodRequest req3 = new MinimumPeriodRequest();
        req3.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "2", "x1"));
        MinimumPeriodRequest req4 = new MinimumPeriodRequest();
        req4.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "3", "x1"));
        MinimumPeriodRequest reqNull = new MinimumPeriodRequest();
        reqNull.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "3", "nonsense"));
        MinimumPeriodRequest reqX3 = new MinimumPeriodRequest();
        reqX3.setLowerLimit(new Attribute(Attribute.AttributeType.numeric, "0", "x3"));

        AttributeRequirement resultBigStarting0 = central.determineMinimumPeriodCentral(req1);
        AttributeRequirement resultBigStarting1 = central.determineMinimumPeriodCentral(req2);
        AttributeRequirement resultBigStarting2 = central.determineMinimumPeriodCentral(req3);
        AttributeRequirement resultBigStarting3 = central.determineMinimumPeriodCentral(req4);
        AttributeRequirement resultBigStartingNull = central.determineMinimumPeriodCentral(reqNull);
        AttributeRequirement resultBigStartingX3 = central.determineMinimumPeriodCentral(reqX3);

        //Small dataset, so entire range is included
        assertEquals(resultSmall.getLowerLimit().getValue(), "0");
        assertEquals(resultSmall.getUpperLimit().getValue(), "inf");
        assertNull(resultSmall.getValue());

        //only 1 example has value 0, so minimum requirement will be [0-2)
        assertEquals(resultBigStarting0.getLowerLimit().getValue(), "0");
        assertEquals(resultBigStarting0.getUpperLimit().getValue(), "2");
        assertNull(resultBigStarting0.getValue());
        //13 examples have value 1, so value will be used, not range
        assertEquals(resultBigStarting1.getValue().getValue(), "1");
        assertNull(resultBigStarting1.getUpperLimit());
        assertNull(resultBigStarting1.getLowerLimit());
        //5 examples have value 2, there are also not enough examples >2, so minimum requirement will be [2-inf]
        assertEquals(resultBigStarting2.getLowerLimit().getValue(), "2");
        assertEquals(resultBigStarting2.getUpperLimit().getValue(), "inf");
        assertNull(resultBigStarting2.getValue());
        //1 example has value 3, so minimum requirement will be [3-inf)
        assertEquals(resultBigStarting3.getLowerLimit().getValue(), "3");
        assertEquals(resultBigStarting3.getUpperLimit().getValue(), "inf");
        assertNull(resultBigStarting3.getValue());

        //This attribute does not exist so the result is null
        assertNull(resultBigStartingNull);

        //x3 only has less than 10 examples with value 0 so the entire range is used again
        assertEquals(resultBigStartingX3.getLowerLimit().getValue(), "0");
        assertEquals(resultBigStartingX3.getUpperLimit().getValue(), "inf");
        assertNull(resultBigStartingX3.getValue());
    }

    @Test
    public void testValuesMultiplicationHybridSplit()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/hybridsplit/smallK2Example_firsthalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/hybridsplit/smallK2Example_secondhalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        VerticoxServer server3 = new VerticoxServer(
                "resources/hybridsplit/smallK2Example_secondhalf_morepopulation.csv", "3");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        BigDecimal[] values = generateValues();
        endpointZ.setValues(createSetValuesRequest(endpointZ, values));

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
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationThreeServers()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);
        VerticoxServer server3 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint3 = new VerticoxEndpoint(server3);

        BigDecimal[] values = generateValues();
        endpointZ.setValues(createSetValuesRequest(endpointZ, values));

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
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        endpoint3.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationInfiniteRange()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = generateValues();
        endpointZ.setValues(createSetValuesRequest(endpointZ, values));

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
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    @Test
    public void testValuesMultiplicationDataAndCriteriaInSamePlace()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "Z");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        BigDecimal[] values = generateValues();
        endpointZ.setValues(createSetValuesRequest(endpointZ, values));

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
        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);

        int precision = 5;
        BigDecimal multiplier = BigDecimal.valueOf(Math.pow(10, precision));
        endpointZ.setPrecision(precision);
        endpoint2.setPrecision(precision);
        central.setPrecisionCentral(precision);
        secret.setPrecision(precision);

        central.initEndpoints(Arrays.asList(endpointZ, endpoint2), secretEnd);
        SumRelevantValuesRequest request = new SumRelevantValuesRequest();
        request.setValueServer("z");
        request.setRequirements(Arrays.asList(req));
        BigDecimal result = central.sumRelevantValues(request);

        assertEquals(result.longValue(), expected.longValue(), multiplier.longValue());
    }

    private SetValuesRequest createSetValuesRequest(VerticoxEndpoint endpointZ, BigDecimal[] values) {
        SetValuesRequest req = new SetValuesRequest();
        try {
            AES aes = new AES();
            byte[] rsaPublicKey = endpointZ.getPublicKey().getKey();
            X509EncodedKeySpec keySpec = new X509EncodedKeySpec(rsaPublicKey);
            KeyFactory keyFactory = KeyFactory.getInstance("RSA");
            PublicKey pubKey = keyFactory.generatePublic(keySpec);
            cipher.init(Cipher.ENCRYPT_MODE, pubKey);
            String[] encrypted = new String[values.length];
            for (int i = 0; i < values.length; i++) {
                encrypted[i] = aes.encrypt(values[i]);
            }
            req.setValues(encrypted);
            req.setEncryptedAes(cipher.doFinal(aes.getKey().getEncoded()));
        } catch (Exception e) {

        }
        return req;
    }
}