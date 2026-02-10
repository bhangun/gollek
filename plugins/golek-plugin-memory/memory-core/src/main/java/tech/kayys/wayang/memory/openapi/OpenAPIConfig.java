package tech.kayys.wayang.memory.openapi;

import org.eclipse.microprofile.openapi.annotations.OpenAPIDefinition;
import org.eclipse.microprofile.openapi.annotations.info.Contact;
import org.eclipse.microprofile.openapi.annotations.info.Info;
import org.eclipse.microprofile.openapi.annotations.info.License;
import org.eclipse.microprofile.openapi.annotations.servers.Server;

import jakarta.ws.rs.core.Application;

@OpenAPIDefinition(
    info = @Info(
        title = "Memory Service API",
        version = "1.0.0",
        description = "Enterprise-grade memory management service for AI agent systems",
        contact = @Contact(
            name = "DevOps Team",
            email = "devops@enterprise.com"
        ),
        license = @License(
            name = "Enterprise License",
            url = "https://enterprise.com/license"
        )
    ),
    servers = {
        @Server(url = "http://localhost:8080", description = "Local Development"),
        @Server(url = "https://memory-service.staging.enterprise.com", description = "Staging"),
        @Server(url = "https://memory-service.enterprise.com", description = "Production")
    }
)
public class OpenAPIConfig extends Application {
}