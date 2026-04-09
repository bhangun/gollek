package tech.kayys.gollek.ml.data;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * A dataset that loads CSV/TSV files with column mapping.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var dataset = new CsvDataset(Path.of("data.csv"), ",", true);
 * Map<String, String> row = dataset.get(0);
 * String label = row.get("label");
 * }</pre>
 */
public class CsvDataset implements Dataset<Map<String, String>> {

    private final List<Map<String, String>> rows;

    public CsvDataset(Path file, String delimiter, boolean hasHeader) throws IOException {
        this.rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(file);
        if (lines.isEmpty()) return;

        String[] headers;
        int startIdx;
        if (hasHeader) {
            headers = lines.get(0).split(delimiter);
            startIdx = 1;
        } else {
            int numCols = lines.get(0).split(delimiter).length;
            headers = new String[numCols];
            for (int i = 0; i < numCols; i++) headers[i] = "col_" + i;
            startIdx = 0;
        }

        for (int i = startIdx; i < lines.size(); i++) {
            String[] values = lines.get(i).split(delimiter, -1);
            Map<String, String> row = new LinkedHashMap<>();
            for (int j = 0; j < headers.length && j < values.length; j++) {
                row.put(headers[j].trim(), values[j].trim());
            }
            rows.add(row);
        }
    }

    @Override
    public Map<String, String> get(int index) {
        return rows.get(index);
    }

    @Override
    public int size() {
        return rows.size();
    }

    /** Get a specific column across all rows. */
    public List<String> column(String name) {
        return rows.stream().map(r -> r.getOrDefault(name, "")).toList();
    }
}
