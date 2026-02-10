package tech.kayys.wayang.memory.ratelimit;

import io.quarkus.redis.datasource.RedisDataSource;
import io.quarkus.redis.datasource.string.StringCommands;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.time.Duration;

@ApplicationScoped
public class RateLimiter {
    
    @Inject
    RedisDataSource redisDataSource;

    public Uni<Boolean> checkRateLimit(String userId, int maxRequests, Duration window) {
        StringCommands<String, String> commands = redisDataSource.string(String.class);
        String key = "ratelimit:" + userId;
        
        return Uni.createFrom().item(() -> {
            String countStr = commands.get(key);
            int count = countStr != null ? Integer.parseInt(countStr) : 0;
            
            if (count >= maxRequests) {
                return false;
            }
            
            if (count == 0) {
                commands.setex(key, window.toSeconds(), "1");
            } else {
                commands.incr(key);
            }
            
            return true;
        });
    }

    public Uni<RateLimitInfo> getRateLimitInfo(String userId) {
        StringCommands<String, String> commands = redisDataSource.string(String.class);
        String key = "ratelimit:" + userId;
        
        return Uni.createFrom().item(() -> {
            String countStr = commands.get(key);
            int count = countStr != null ? Integer.parseInt(countStr) : 0;
            
            return new RateLimitInfo(userId, count);
        });
    }

    public static class RateLimitInfo {
        public final String userId;
        public final int currentRequests;

        public RateLimitInfo(String userId, int currentRequests) {
            this.userId = userId;
            this.currentRequests = currentRequests;
        }
    }
}