package tech.kayys.gollek.ml.data;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Simple text dataset that loads lines from a file.
 */
public class TextDataset implements Dataset<String> {

    private final List<String> lines;

    public TextDataset(Path file) throws IOException {
        this.lines = Files.readAllLines(file);
    }

    @Override
    public String get(int index) {
        return lines.get(index);
    }

    @Override
    public int size() {
        return lines.size();
    }
}
