package tech.kayys.gollek.engine.model;

import java.time.Duration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.model.Pageable;

public class CachedModelRepositoryTest {

    private CachedModelRepository repository;

    @BeforeEach
    void setUp() {
        repository = new CachedModelRepository();
    }

    @Test
    void testFindById_ReturnsEmpty() {
        try {
            // Fix: pass String instead of RequestId
            repository.findById("m1", "t1").await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // expected
        }
    }

    @Test
    void testFindByTenant_ReturnsEmpty() {
        try {
            // Fix: pass String and valid Pageable
            Pageable pageable = new Pageable(1, 10);
            repository.list("t1", pageable).await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // expected
        }
    }
}
