package com.florian.verticox.webservice;

import com.florian.nscalarproduct.webservice.Server;
import com.florian.nscalarproduct.webservice.ServerEndpoint;
import com.florian.verticox.webservice.domain.InitDataResponse;
import com.florian.verticox.webservice.domain.SumPredictorInTimeFrameRequest;

import java.math.BigDecimal;

public class VerticoxEndpoint extends ServerEndpoint {
    public VerticoxEndpoint(Server server) {
        super(server);
    }

    public VerticoxEndpoint(String url) {
        super(url);
    }

    public BigDecimal getSum() {
        if (testing) {
            return ((VerticoxServer) (server)).getSum();
        } else {
            return REST_TEMPLATE.getForEntity(serverUrl + "/getSum", BigDecimal.class).getBody();
        }
    }

    public InitDataResponse initData(SumPredictorInTimeFrameRequest req) {
        if (testing) {
            return ((VerticoxServer) (server)).initData(req);
        } else {
            return REST_TEMPLATE.postForEntity(serverUrl + "/initData", req, InitDataResponse.class).getBody();
        }
    }

    public void setPrecision(int precision) {
        if (testing) {
            ((VerticoxServer) (server)).setPrecision(precision);
        } else {
            REST_TEMPLATE.put(serverUrl + "/setPrecision?precision=" + precision, Void.class);
        }
    }

}
