package tech.kayys.wayang.memory.filter;

import tech.kayys.wayang.memory.ratelimit.RateLimiter;
import jakarta.inject.Inject;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.Provider;

import java.time.Duration;

@Provider
public class RateLimitFilter implements ContainerRequestFilter {
    
    @Inject
    RateLimiter rateLimiter;

    @Override
    public void filter(ContainerRequestContext requestContext) {
        String userId = requestContext.getHeaderString("X-User-Id");
        
        if (userId != null) {
            Boolean allowed = rateLimiter.checkRateLimit(
                userId, 
                100, // 100 requests
                Duration.ofMinutes(1) // per minute
            ).await().indefinitely();
            
            if (!allowed) {
                requestContext.abortWith(
                    Response.status(Response.Status.TOO_MANY_REQUESTS)
                        .entity("{\"error\": \"Rate limit exceeded\"}")
                        .build()
                );
            }
        }
    }
}