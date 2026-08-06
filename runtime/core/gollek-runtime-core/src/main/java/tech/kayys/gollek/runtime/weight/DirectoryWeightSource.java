package tech.kayys.gollek.runtime.weight;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.alkhawarizm.core.memory.CpuBuffer;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A weight source that loads weights from individual files in a directory.
 */
public final class DirectoryWeightSource implements WeightSource {
    private final Path directory;
    private final ComputeBackend backend;
    private final String extension;

    public DirectoryWeightSource(Path directory, ComputeBackend backend) {
        this(directory, backend, ".bin");
    }

    public DirectoryWeightSource(Path directory, ComputeBackend backend, String extension) {
        this.directory = directory;
        this.backend = backend;
        this.extension = extension;
    }

    @Override
    public Tensor get(String key) {
        Path path = directory.resolve(key + extension);
        if (!Files.exists(path)) {
            return null;
        }

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            // Simple format: [rank:int][dims:long...][dtype:int][data...]
            ByteBuffer header = ByteBuffer.allocate(4);
            channel.read(header);
            header.flip();
            int rank = header.getInt();

            ByteBuffer dimsBuf = ByteBuffer.allocate(rank * 8);
            channel.read(dimsBuf);
            dimsBuf.flip();
            long[] dims = new long[rank];
            for (int i = 0; i < rank; i++) {
                dims[i] = dimsBuf.getLong();
            }

            ByteBuffer dtypeBuf = ByteBuffer.allocate(4);
            channel.read(dtypeBuf);
            dtypeBuf.flip();
            int dtypeOrdinal = dtypeBuf.getInt();
            DType dtype = DType.values()[dtypeOrdinal];

            long dataSize = channel.size() - channel.position();
            CpuBuffer buffer = new CpuBuffer(dataSize);
            ByteBuffer dest = buffer.segment().asByteBuffer();
            while (dest.hasRemaining()) {
                if (channel.read(dest) == -1) break;
            }

            return new DefaultTensor(new Shape(dims), dtype, DeviceType.CPU, buffer, backend);
        } catch (IOException e) {
            throw new InferenceException(ErrorCode.INTERNAL_ERROR, "Failed to load weight: " + key, e);
        }
    }

    @Override
    public boolean contains(String key) {
        return Files.exists(directory.resolve(key + extension));
    }

    @Override
    public Set<String> keys() {
        try (Stream<Path> stream = Files.list(directory)) {
            return stream
                    .filter(p -> p.toString().endsWith(extension))
                    .map(p -> {
                        String name = p.getFileName().toString();
                        return name.substring(0, name.length() - extension.length());
                    })
                    .collect(Collectors.toSet());
        } catch (IOException e) {
            return new HashSet<>();
        }
    }
}
