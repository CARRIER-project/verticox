package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.verticox.webservice.domain.InitZRequest;
import com.florian.verticox.webservice.domain.SumZRequest;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.NoSuchPaddingException;
import java.io.UnsupportedEncodingException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VerticoxEndpointZTest {

    private Cipher cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");

    public VerticoxEndpointZTest() throws NoSuchPaddingException, NoSuchAlgorithmException {
    }

    @Test
    public void testValuesMultiplication_ZAndOutComeLocal()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "1");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        //this selects individuals: 1,2,4,7, & 9 = 1+3+6+8=18

        double[] z = new double[10];
        for (int i = 0; i < 10; i++) {
            z[i] = i;
        }


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

        InitZRequest request = new InitZRequest();
        request.setZ(z);
        request.setAttribute("x1");

        central.postZ(request);

        SumZRequest request2 = new SumZRequest();
        request2.setRequirements(Arrays.asList(req));


        double result = central.sumZValues(request2);
        int expected = 18;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplication_ZRequirementRange()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "1");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        Attribute upper = new Attribute(Attribute.AttributeType.numeric, "10", "x1");
        Attribute lower = new Attribute(Attribute.AttributeType.numeric, "1", "x1");
        AttributeRequirement req = new AttributeRequirement(lower, upper);
        //this selects individuals: 1,2,4,7, & 9 = 1+3+6+8=18

        double[] z = new double[10];
        for (int i = 0; i < 10; i++) {
            z[i] = i;
        }


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

        InitZRequest request = new InitZRequest();
        request.setZ(z);
        request.setAttribute("x1");


        central.postZ(request);

        SumZRequest request2 = new SumZRequest();
        request2.setRequirements(Arrays.asList(req));


        double result = central.sumZValues(request2);
        int expected = 18;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplication_ZandRequirementSplit()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "1");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        Attribute lower = new Attribute(Attribute.AttributeType.numeric, "1", "x3");
        AttributeRequirement req = new AttributeRequirement(lower);
        //this selects individuals: 2,3,4,6,7,9 = 1+2+3+5+6+8=25

        double[] z = new double[10];
        for (int i = 0; i < 10; i++) {
            z[i] = i;
        }


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

        InitZRequest request = new InitZRequest();
        request.setZ(z);
        request.setAttribute("x3");


        central.postZ(request);


        SumZRequest request2 = new SumZRequest();
        request2.setRequirements(Arrays.asList(req));


        double result = central.sumZValues(request2);
        int expected = 25;

        assertEquals(result, expected);
    }
    
    @Test
    public void testValuesMultiplication_ZMultipleRequirements()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "1");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        AttributeRequirement req2 = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        req2.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x2"));
        //this selects individuals: 2,4,7, & 9 = 1 + 3 + 6 + 8 = 18


        double[] z = new double[10];
        for (int i = 0; i < 10; i++) {
            z[i] = i;
        }


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

        InitZRequest request = new InitZRequest();
        request.setZ(z);
        request.setAttribute("x2");

        central.postZ(request);

        SumZRequest request2 = new SumZRequest();
        request2.setRequirements(Arrays.asList(req, req2));


        double result = central.sumZValues(request2);
        int expected = 18;

        assertEquals(result, expected);
    }

    @Test
    public void testValuesMultiplication_ZSplitRequirements()
            throws NoSuchPaddingException, UnsupportedEncodingException, NoSuchAlgorithmException {
        VerticoxServer serverZ = new VerticoxServer("resources/smallK2Example_secondhalf.csv", "1");
        VerticoxEndpoint endpointZ = new VerticoxEndpoint(serverZ);

        VerticoxServer server2 = new VerticoxServer("resources/smallK2Example_firsthalf.csv", "2");
        VerticoxEndpoint endpoint2 = new VerticoxEndpoint(server2);

        AttributeRequirement req = new AttributeRequirement();
        AttributeRequirement req2 = new AttributeRequirement();
        req.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x1"));
        req2.setValue(new Attribute(Attribute.AttributeType.numeric, "1", "x3"));
        //this selects individuals: 4,7, & 9 =  3 + 6 + 8 = 17


        double[] z = new double[10];
        for (int i = 0; i < 10; i++) {
            z[i] = i;
        }


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

        InitZRequest request = new InitZRequest();
        request.setZ(z);
        request.setAttribute("x3");

        central.postZ(request);

        SumZRequest request2 = new SumZRequest();
        request2.setRequirements(Arrays.asList(req, req2));


        double result = central.sumZValues(request2);
        int expected = 18;

        assertEquals(result, expected);
    }
}