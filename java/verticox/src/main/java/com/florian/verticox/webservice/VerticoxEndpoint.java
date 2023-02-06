package com.florian.verticox.webservice;

import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.verticox.webservice.domain.*;

public class VerticoxEndpoint extends ServerEndpoint {
    public VerticoxEndpoint(Server server) {
        super(server);
    }

    public VerticoxEndpoint(String url) {
        super(url);
    }

    public Double getSum() {
        if (testing) {
            return ((VerticoxServer) (server)).getSum();
        } else {
            return REST_TEMPLATE.getForEntity(serverUrl + "/getSum", Double.class).getBody();
        }
    }

    public Integer getCount() {
        if (testing) {
            return ((VerticoxServer) (server)).getCount();
        } else {
            return REST_TEMPLATE.getForEntity(serverUrl + "/getCount", Integer.class).getBody();
        }
    }

    public InitDataResponse initData(SumPredictorInTimeFrameRequest req) {
        if (testing) {
            return ((VerticoxServer) (server)).initData(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/initData", req, InitDataResponse.class).getBody();
        }
    }

    public boolean containsAttribute(InitZRequest req) {
        if (testing) {
            return ((VerticoxServer) (server)).containsAttribute(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/containsAttribute", req, Boolean.class).getBody();
        }
    }

    public UniqueValueResponse getUniqueValues(RelevantValueRequest req) {
        if (testing) {
            return ((VerticoxServer) (server)).getUniqueValues(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/getUniqueValues", req, UniqueValueResponse.class)
                    .getBody();
        }
    }

    public InitDataResponse initRt(SumZRequest req) {
        if (testing) {
            return ((VerticoxServer) (server)).initRt(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/initRt", req, InitDataResponse.class).getBody();
        }
    }

    public void initZData(double[] z) {
        if (testing) {
            ((VerticoxServer) (server)).initZData(z);
        } else {
            REST_TEMPLATE.put(serverUrl + "/initZData", z);
        }
    }

    public void setPrecision(int precision) {
        if (testing) {
            ((VerticoxServer) (server)).setPrecision(precision);
        } else {
            REST_TEMPLATE.put(serverUrl + "/setPrecision?precision=" + precision, Void.class);
        }
    }

    public void setActiveRecords(ActiveRecordRequest request) {
        if (testing) {
            ((VerticoxServer) (server)).setActiveRecords(request);
        } else {
            REST_TEMPLATE.postForEntity(serverUrl + "/setActiveRecords", request, void.class);
        }
    }
}
