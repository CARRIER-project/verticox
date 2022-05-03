package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirementsRequest;
import com.florian.verticox.webservice.domain.MinimumPeriodRequest;
import com.florian.verticox.webservice.domain.PublicKeyResponse;
import com.florian.verticox.webservice.domain.SetValuesRequest;

import javax.crypto.NoSuchPaddingException;
import java.security.NoSuchAlgorithmException;
import java.util.List;

public class VerticoxEndpoint extends ServerEndpoint {
    public VerticoxEndpoint(Server server) {
        super(server);
    }

    public VerticoxEndpoint(String url) {
        super(url);
    }

    public void setValues(SetValuesRequest req) throws NoSuchPaddingException, NoSuchAlgorithmException {
        if (testing) {
            ((VerticoxServer) (server)).setValues(req);
        } else {
            REST_TEMPLATE.put(serverUrl + "/setZValues", req);
        }
    }

    public PublicKeyResponse getPublicKey() {
        if (testing) {
            return ((VerticoxServer) (server)).getPublicKey();
        } else {
            return REST_TEMPLATE.getForEntity(serverUrl + "/getPublicKey", PublicKeyResponse.class).getBody();
        }
    }


    public void initData(List<AttributeRequirement> requirements) {
        AttributeRequirementsRequest req = new AttributeRequirementsRequest();
        req.setRequirements(requirements);
        if (testing) {
            ((VerticoxServer) (server)).initValueData(req);
        } else {
            REST_TEMPLATE.put(serverUrl + "/initData", req, Void.class);
        }
    }

    public void setPrecision(int precision) {
        if (testing) {
            ((VerticoxServer) (server)).setPrecision(precision);
        } else {
            REST_TEMPLATE.put(serverUrl + "/setPrecision?precision=" + precision, Void.class);
        }
    }

    public AttributeRequirement determineMinimumPeriod(Attribute lower) {
        MinimumPeriodRequest req = new MinimumPeriodRequest();
        req.setLowerLimit(lower);
        if (testing) {
            return ((VerticoxServer) (server)).determineMinimumPeriod(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/determineMinimumPeriod", req, AttributeRequirement.class)
                    .getBody();
        }
    }

    public void selectIndividuals(List<AttributeRequirement> requirements) {
        AttributeRequirementsRequest req = new AttributeRequirementsRequest();
        req.setRequirements(requirements);
        if (testing) {
            ((VerticoxServer) (server)).selectIndividuals(req);
        } else {
            REST_TEMPLATE.put(serverUrl + "/selectIndividuals", req);
        }
    }

}
