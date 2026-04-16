package tech.kayys.gollek.cache;

import jakarta.inject.Qualifier;
import java.lang.annotation.Documented;
import java.lang.annotation.Retention;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.*;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

/**
 * CDI qualifier that selects the {@link PromptCacheStore} implementation
 * matching a specific storage strategy.
 *
 * <p>Usage on an implementation:
 * <pre>{@code
 * @ApplicationScoped
 * @CacheStrategy("redis")
 * public class RedisPromptCacheStore implements PromptCacheStore { ... }
 * }</pre>
 *
 * <p>Valid values: {@code "in-process"}, {@code "redis"}, {@code "disk"}, {@code "noop"}.
 */
@Qualifier
@Documented
@Retention(RUNTIME)
@Target({TYPE, METHOD, FIELD, PARAMETER})
public @interface CacheStrategy {
    String value();
}
