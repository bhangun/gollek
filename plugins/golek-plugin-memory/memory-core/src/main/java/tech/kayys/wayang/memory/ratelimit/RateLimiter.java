package tech.kayys.wayang.memory.ratelimit;

import io.quarkus.redis.datasource.RedisDataSource;
import io.quarkus.redis.datasource.string.StringCommands;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;

import java.time.Duration;

@ApplicationScoped
public class RateLimiter {
    
    @Inject
    Instance<RedisDataSource> redisDataSource;

    public Uni<Boolean> checkRateLimit(String userId, int maxRequests, Duration window) {
        if (!redisDataSource.isResolvable()) {
            return Uni.createFrom().item(true); // Fail open if Redis unavailable
        }

        return Uni.createFrom().item(() -> {
            try {
                RedisDataSource ds = redisDataSource.get();
                StringCommands<String, String> commands = ds.string(String.class);
                String key = "ratelimit:" + userId;
                
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
            } catch (Exception e) {
                return true; // Fail open
            }
        });
    }

    public Uni<RateLimitInfo> getRateLimitInfo(String userId) {
        if (!redisDataSource.isResolvable()) {
            return Uni.createFrom().item(new RateLimitInfo(userId, 0));
        }

        return Uni.createFrom().item(() -> {
            try {
                RedisDataSource ds = redisDataSource.get();
                StringCommands<String, String> commands = ds.string(String.class);
                String key = "ratelimit:" + userId;
                
                String countStr = commands.get(key);
                int count = countStr != null ? Integer.parseInt(countStr) : 0;
                
                return new RateLimitInfo(userId, count);
            } catch (Exception e) {
                return new RateLimitInfo(userId, 0);
            }
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