# Database Configuration Guide

This guide explains how to configure the database for different deployment modes in the Gollek platform.

## Overview

The Gollek platform supports two deployment modes with different database configurations:

| Mode | Database | Use Case |
|------|----------|----------|
| **Standalone/Demo** | H2 (in-memory) | Local development, testing, demos |
| **Enterprise** | PostgreSQL (reactive) | Production, distributed runtime |

## Quick Start

### Standalone/Demo Mode (H2)

```bash
# Using the standalone runtime (default: H2)
cd runtime/gollek-runtime-standalone
./mvnw quarkus:dev

# Or using the unified runtime with standalone profile
cd runtime/gollek-runtime-unified
./mvnw quarkus:dev -Dquarkus.profile=standalone
```

No database setup required! H2 runs in-memory.

### Enterprise Mode (PostgreSQL)

```bash
# Dev mode with auto-provisioned PostgreSQL (Dev Services - DEFAULT)
cd runtime/gollek-runtime-unified
./mvnw quarkus:dev

# Production mode with existing PostgreSQL
export DB_USERNAME=postgres
export DB_PASSWORD=your-password
export DATABASE_URL=postgresql://localhost:5432/gollek
./mvnw quarkus:dev -Dquarkus.profile=prod
```

## Configuration Options

### Environment Variables (Production Mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_USERNAME` | `postgres` | Database username |
| `DB_PASSWORD` | `postgres` | Database password |
| `DATABASE_URL` | `postgresql://localhost:5432/gollek` | PostgreSQL connection URL |
| `DB_POOL_MAX_SIZE` | `20` | Maximum connection pool size |
| `DB_POOL_MIN_SIZE` | `5` | Minimum connection pool size |
| `HIBERNATE_DB_GENERATION` | `validate` | Schema generation strategy |

### Application Profiles

| Runtime | Profile | Database | Schema Generation |
|---------|---------|----------|-------------------|
| `gollek-runtime-unified` | `dev` (default) | PostgreSQL (Dev Services) | `drop-and-create` |
| `gollek-runtime-unified` | `prod` | PostgreSQL (manual) | `validate` |
| `gollek-runtime-unified` | `standalone` | H2 | `drop-and-create` |
| `gollek-runtime-unified` | `demo` | H2 | `drop-and-create` |
| `gollek-runtime-standalone` | default | H2 | `drop-and-create` |
| `gollek-runtime-standalone` | `prod` | PostgreSQL (manual) | `validate` |

## Dev Services

Quarkus Dev Services automatically provisions a PostgreSQL container in development mode (default for `gollek-runtime-unified`):

```bash
# Automatically starts PostgreSQL container
cd runtime/gollek-runtime-unified
./mvnw quarkus:dev

# Use H2 instead (no container needed)
./mvnw quarkus:dev -Dquarkus.profile=standalone
```

### Dev Services Configuration

Dev Services is enabled by default in dev mode. To customize:

```properties
# Custom database name
quarkus.datasource.devservices.db-name=gollek

# Disable Dev Services (use existing database)
quarkus.datasource.devservices.enabled=false
```

## Database Schema

The platform uses Hibernate ORM with the following entities:

- `Model` - Model registry and metadata
- `ModelVersion` - Model versioning
- `InferenceRequestEntity` - Inference request tracking

### Schema Generation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `drop-and-create` | Drops and recreates schema on each startup | Development, testing |
| `create` | Creates schema on startup | Development |
| `validate` | Validates schema against entities | Production |
| `none` | No schema management | Manual schema management |

## PostgreSQL Setup

### Using Docker

```bash
docker run -d \
  --name gollek-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=gollek \
  -p 5432:5432 \
  postgres:15
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: gollek
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

```bash
docker-compose up -d
```

### Connection String Format

```
postgresql://<host>:<port>/<database>
```

Example:
```
postgresql://localhost:5432/gollek
```

## Troubleshooting

### "No pool has been defined" Error

This error occurs when Hibernate ORM entities are discovered but no datasource is configured.

**Solution:**
1. Ensure database dependencies are in `pom.xml`:
   ```xml
   <dependency>
       <groupId>io.quarkus</groupId>
       <artifactId>quarkus-reactive-pg-client</artifactId>
   </dependency>
   <dependency>
       <groupId>io.quarkus</groupId>
       <artifactId>quarkus-hibernate-reactive-panache</artifactId>
   </dependency>
   ```

2. Configure datasource in `application.properties`

3. For H2 mode, set:
   ```bash
   export QUARKUS_DATASOURCE_DB_KIND=h2
   ```

### Dev Services Not Starting

**Check:**
- Docker is running
- Port 5432 is not in use
- `quarkus.datasource.devservices.enabled=true`

### Connection Issues

**Verify:**
```bash
# Test PostgreSQL connection
psql -h localhost -U postgres -d gollek

# Check if PostgreSQL is running
docker ps | grep postgres
```

## Migration Guide

### From Standalone to Enterprise

1. Update environment variables:
   ```bash
   export QUARKUS_DATASOURCE_DB_KIND=postgresql
   export DATABASE_URL=postgresql://localhost:5432/gollek
   ```

2. Set schema generation to `validate`:
   ```bash
   export HIBERNATE_DB_GENERATION=validate
   ```

3. Run with production profile:
   ```bash
   ./mvnw quarkus:dev -Dquarkus.profile=prod
   ```

## Performance Tuning

### Connection Pool Settings

```properties
# Adjust based on workload
quarkus.datasource.reactive.max-size=20
quarkus.datasource.reactive.min-size=5

# Connection timeout (milliseconds)
quarkus.datasource.reactive.acquire-retry=3
quarkus.datasource.reactive.connection-timeout=30000
```

### Query Logging

```bash
# Enable SQL logging for debugging
export HIBERNATE_LOG_SQL=true
export HIBERNATE_SHOW_SQL=true
```

## Security Considerations

- **Never** commit database credentials to version control
- Use environment variables or secure vault for credentials
- Enable SSL for production PostgreSQL connections
- Use strong passwords and rotate regularly
- Limit database user permissions to minimum required

## Additional Resources

- [Quarkus Hibernate Reactive Guide](https://quarkus.io/guides/hibernate-reactive)
- [Quarkus Dev Services](https://quarkus.io/guides/dev-services)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
