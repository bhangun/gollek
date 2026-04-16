package tech.kayys.gollek.provider.cerebras;

/* 
 * Obsolete REST client - replaced by tech.kayys.gollek.provider.cerebras.CerebrasProvider 
 * using java.net.http.HttpClient for better compatibility.
 *
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@RegisterRestClient(configKey = "cerebras-api", baseUri = "https://api.cerebras.ai")
@Path("/v1")
public interface CerebrasClient {
...
}
*/
public interface CerebrasClient {
        // Empty interface to satisfy dependencies if any, though none should remain.
}