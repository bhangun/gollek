# Dockerfile
FROM registry.access.redhat.com/ubi8/openjdk-21-runtime:1.17

COPY target/quarkus-app/lib/ /deployments/lib/
COPY target/quarkus-app/*.jar /deployments/
COPY target/quarkus-app/app/ /deployments/app/
COPY target/quarkus-app/quarkus/ /deployments/quarkus/

EXPOSE 8080

USER 185

ENTRYPOINT ["java", "-jar", "/deployments/quarkus-run.jar"]