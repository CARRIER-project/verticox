package com.florian.verticox.webservice;

import com.florian.nscalarproduct.data.Data;
import com.florian.nscalarproduct.webservice.Server;
import org.springframework.web.bind.annotation.PutMapping;

import java.math.BigDecimal;
import java.math.BigInteger;

import static com.florian.nscalarproduct.data.Parser.parseCsv;

public class VerticoxServer extends Server {
    private Data data;
    private BigDecimal[] zValues;
    private String path;

    @PutMapping ("setZValues")
    public void setZValues(BigDecimal[] zValues) {
        this.zValues = zValues;
    }

    @PutMapping ("initZData")
    public void initZData() {
        reset();
        localData = new BigInteger[population];
        for (int i = 0; i < population; i++) {
            //Needs to work with decimals, original n-party implementation only uses integers
            localData[i] = zValues[i];
        }
    }

    private void readData() {
        if (System.getenv("DATABASE_URI") != null) {
            // Check if running in vantage6 by looking for system env, if yes change to database_uri system env for path
            this.path = System.getenv("DATABASE_URI");
        }
        this.data = parseCsv(path, 0);
    }

}
