package tech.kayys.gollek.converter.gguf;

import java.util.Arrays;

/**
 * Descriptor for a single tensor inside a GGUF file.
 * Mirrors {@code gguf_tensor_info} from the C spec.
 *
 * <p>
 * Dimensions are stored exactly as on-disk: {@code ne[0]} is the
 * innermost (fastest-varying) dimension.
 */
public record TensorInfo(
        String name,
        long[] ne, // shape; length == nDims, padded to 4 with 1s internally
        GgmlType type,
        long offset // byte offset from start of tensor-data section
) {

    public TensorInfo {
        if (name == null || name.isBlank())
            throw new IllegalArgumentException("name");
        if (ne == null || ne.length == 0 || ne.length > 4)
            throw new IllegalArgumentException("ne must have 1-4 dimensions");
        for (long d : ne)
            if (d <= 0)
                throw new IllegalArgumentException("dimension <= 0");
        if (type == null)
            throw new IllegalArgumentException("type");
        if (offset < 0)
            throw new IllegalArgumentException("offset < 0");
    }

    /** Total element count (product of all dimensions). */
    public long numElements() {
        long n = 1;
        for (long d : ne)
            n *= d;
        return n;
    }

    /** On-disk byte footprint for this tensor's data. */
    public long dataSize() {
        return type.bytesFor(numElements());
    }

    /** Number of dimensions stored in {@link #ne()}. */
    public int nDims() {
        return ne.length;
    }

    @Override
    public String toString() {
        return "TensorInfo{name='%s', shape=%s, type=%s, offset=%d}"
                .formatted(name, Arrays.toString(ne), type.label, offset);
    }
}
