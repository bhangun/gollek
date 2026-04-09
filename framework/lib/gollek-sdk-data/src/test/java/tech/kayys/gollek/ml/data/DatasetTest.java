package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class DatasetTest {

    @TempDir
    Path tempDir;

    @Test
    public void testCsvDatasetWithHeader() throws IOException {
        Path csvFile = tempDir.resolve("test.csv");
        Files.writeString(csvFile, "id,name,value\n1,alpha,10.0\n2,beta,20.0");

        CsvDataset dataset = new CsvDataset(csvFile, ",", true);
        assertEquals(2, dataset.size());

        Map<String, String> row1 = dataset.get(0);
        assertEquals("1", row1.get("id"));
        assertEquals("alpha", row1.get("name"));
        assertEquals("10.0", row1.get("value"));

        List<String> values = dataset.column("value");
        assertEquals(List.of("10.0", "20.0"), values);
    }

    @Test
    public void testCsvDatasetNoHeader() throws IOException {
        Path csvFile = tempDir.resolve("test_no_header.csv");
        Files.writeString(csvFile, "1,alpha,10.0\n2,beta,20.0");

        CsvDataset dataset = new CsvDataset(csvFile, ",", false);
        assertEquals(2, dataset.size());

        Map<String, String> row1 = dataset.get(0);
        assertEquals("1", row1.get("col_0"));
        assertEquals("alpha", row1.get("col_1"));
        assertEquals("10.0", row1.get("col_2"));
    }

    @Test
    public void testTextDataset() throws IOException {
        Path txtFile = tempDir.resolve("test.txt");
        Files.writeString(txtFile, "line 1\nline 2\nline 3");

        TextDataset dataset = new TextDataset(txtFile);
        assertEquals(3, dataset.size());
        assertEquals("line 1", dataset.get(0));
        assertEquals("line 3", dataset.get(2));
    }
}
