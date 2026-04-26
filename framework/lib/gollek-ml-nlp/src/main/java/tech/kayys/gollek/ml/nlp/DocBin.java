package tech.kayys.gollek.ml.nlp;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * High-performance container for serializing multiple Doc objects.
 * Uses FFM API for memory-mapped storage and zero-copy access.
 */
public class DocBin {
    private final List<Doc> docs = new ArrayList<>();

    public void add(Doc doc) {
        docs.add(doc);
    }

    public List<Doc> getDocs() {
        return docs;
    }

    /**
     * Serializes all docs to a binary file using memory mapping.
     */
    public void toFile(Path path) throws IOException {
        // Calculate total size: [numDocs (long)] + [doc1_size (long)] + [doc1_data] + ...
        long totalSize = ValueLayout.JAVA_LONG.byteSize();
        for (Doc doc : docs) {
            totalSize += ValueLayout.JAVA_LONG.byteSize(); // Size of the doc text
            totalSize += doc.getText().getBytes().length;
            totalSize += ValueLayout.JAVA_INT.byteSize(); // Num tokens
            // ... for a "real" implementation, we'd serialize all Token features, vectors, etc.
        }

        try (var channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalSize, arena);
                
                long offset = 0;
                segment.set(ValueLayout.JAVA_LONG, offset, docs.size());
                offset += ValueLayout.JAVA_LONG.byteSize();

                for (Doc doc : docs) {
                    byte[] textBytes = doc.getText().getBytes();
                    segment.set(ValueLayout.JAVA_LONG, offset, textBytes.length);
                    offset += ValueLayout.JAVA_LONG.byteSize();
                    
                    MemorySegment.copy(MemorySegment.ofArray(textBytes), 0, segment, offset, textBytes.length);
                    offset += textBytes.length;

                    segment.set(ValueLayout.JAVA_INT, offset, doc.length());
                    offset += ValueLayout.JAVA_INT.byteSize();
                }
            }
        }
    }

    /**
     * Loads docs from a binary file using zero-copy memory mapping.
     */
    public static DocBin fromFile(Path path) throws IOException {
        DocBin bin = new DocBin();
        try (var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            try (Arena arena = Arena.ofAuto()) {
                MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
                
                long offset = 0;
                long numDocs = segment.get(ValueLayout.JAVA_LONG, offset);
                offset += ValueLayout.JAVA_LONG.byteSize();

                for (int i = 0; i < numDocs; i++) {
                    long textLen = segment.get(ValueLayout.JAVA_LONG, offset);
                    offset += ValueLayout.JAVA_LONG.byteSize();
                    
                    byte[] textBytes = new byte[(int) textLen];
                    MemorySegment.copy(segment, offset, MemorySegment.ofArray(textBytes), 0, textLen);
                    offset += textLen;

                    Doc doc = new Doc(new String(textBytes));
                    int numTokens = segment.get(ValueLayout.JAVA_INT, offset);
                    offset += ValueLayout.JAVA_INT.byteSize();
                    
                    // In a "real" implementation, we'd reconstruct tokens from the binary stream here
                    bin.add(doc);
                }
            }
        }
        return bin;
    }
}
