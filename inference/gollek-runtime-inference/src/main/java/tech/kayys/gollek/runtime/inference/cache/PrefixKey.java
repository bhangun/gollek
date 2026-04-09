package tech.kayys.gollek.runtime.inference.cache;

import java.util.List;

/**
 * Stable hash key for prefix cache lookups.
 * <p>
 * Two token sequences with the same content produce the same key,
 * enabling O(1) prefix cache lookups.
 */
public record PrefixKey(int hash) {

    public static PrefixKey of(List<Integer> tokens) {
        int h = 1;
        for (int t : tokens) {
            h = 31 * h + t;
        }
        return new PrefixKey(h);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    @Override
    public boolean equals(Object o) {
        return (o instanceof PrefixKey pk) && pk.hash == hash;
    }
}
