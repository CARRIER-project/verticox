package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Attribute;
import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirement;
import com.florian.nscalarproduct.webservice.domain.AttributeRequirementsRequest;
import com.florian.verticox.webservice.domain.MinimumPeriodRequest;

import java.math.BigDecimal;
import java.util.List;

public class VerticoxEndpoint extends ServerEndpoint {
    public VerticoxEndpoint(Server server) {
        super(server);
    }

    public VerticoxEndpoint(String url) {
        super(url);
    }

    public void setValues(BigDecimal[] data) {
        if (testing) {
            ((VerticoxServer) (server)).setValues(data);
        } else {
            REST_TEMPLATE.put(serverUrl + "/setZValues", data);
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
            return REST_TEMPLATE.getForEntity(serverUrl + "/countIndividuals", AttributeRequirement.class, req)
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
