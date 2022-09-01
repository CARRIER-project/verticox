FROM openjdk:15-alpine
ARG JAR_FILE=target/*.jar
COPY ${JAR_FILE} app.jar
COPY DockerFileAPI /resources

ENTRYPOINT ["java","-jar","/app.jar","--spring.config.location=file:/resources/application.properties"]

