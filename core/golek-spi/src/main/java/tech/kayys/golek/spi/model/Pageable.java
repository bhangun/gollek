package tech.kayys.golek.spi.model;

/**
 * Simple pagination parameters.
 */
public record Pageable(int page, int size) {
    public static Pageable of(int page, int size) {
        return new Pageable(page, size);
    }

    public int offset() {
        return page * size;
    }
}
