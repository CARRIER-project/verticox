package com.florian.verticox;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@SuppressWarnings ("checkstyle:HideUtilityClassConstructor")
public class WebserviceDtApplication {
    public static void main(String[] args) {
        //needs to be in com.florian.nscalarproduct to detect the endpoints from the nparty library
        //however, to avoid conflicts don't put it there
        SpringApplication.run(WebserviceDtApplication.class, args);
    }
}
