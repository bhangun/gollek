package tech.kayys.golek.model.core;

/**
 * Simple pagination parameters
 */
public record Pageable(int page, int size) {
    public static Pageable of(int page, int size) {
        return new Pageable(page, size);
    }

    public int offset() {
        return page * size;
    }
}
