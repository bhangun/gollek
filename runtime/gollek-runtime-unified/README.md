# Gollek Unified Runtime

A unified runtime that combines the CLI, REST API, and Web UI in a single executable JAR.

## Features

- Command Line Interface (CLI) for local model management and inference
- REST API for programmatic access to inference capabilities
- Web UI for interactive model management and inference
- Single executable JAR for easy deployment

## Building

```bash
mvn clean package
```

## Running

```bash
java -jar target/gollek-runtime-unified-{version}-runner.jar
```

## Configuration

The unified runtime can be configured via standard Quarkus configuration mechanisms:
- Application properties file (`application.properties`)
- Environment variables
- Command line arguments

## Endpoints

Once running, the service provides:
- REST API at `/api/v1/*`
- Web UI at `/`
- Health checks at `/q/health`
- Metrics at `/q/metrics`
- OpenAPI documentation at `/q/openapi` and `/q/swagger-ui`