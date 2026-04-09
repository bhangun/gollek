package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class DataLoaderTest {

    @Test
    public void testDataLoaderBatching() {
        Dataset<Integer> dataset = new Dataset<Integer>() {
            @Override
            public Integer get(int index) { return index; }
            @Override
            public int size() { return 10; }
        };

        // Batch size 3, don't drop last
        DataLoader<Integer> loader = new DataLoader<>(dataset, 3, false, false);
        assertEquals(4, loader.numBatches());

        List<List<Integer>> batches = new ArrayList<>();
        for (List<Integer> batch : loader) {
            batches.add(batch);
        }

        assertEquals(4, batches.size());
        assertEquals(List.of(0, 1, 2), batches.get(0));
        assertEquals(List.of(3, 4, 5), batches.get(1));
        assertEquals(List.of(6, 7, 8), batches.get(2));
        assertEquals(List.of(9), batches.get(3));
    }

    @Test
    public void testDataLoaderDropLast() {
        Dataset<Integer> dataset = new Dataset<Integer>() {
            @Override
            public Integer get(int index) { return index; }
            @Override
            public int size() { return 10; }
        };

        // Batch size 3, drop last
        DataLoader<Integer> loader = new DataLoader<>(dataset, 3, false, true);
        assertEquals(3, loader.numBatches());

        int count = 0;
        for (List<Integer> batch : loader) {
            assertEquals(3, batch.size());
            count++;
        }
        assertEquals(3, count);
    }

    @Test
    public void testDataLoaderMapping() {
        Dataset<Integer> dataset = new Dataset<Integer>() {
            @Override
            public Integer get(int index) { return index; }
            @Override
            public int size() { return 10; }
        };

        DataLoader<String> loader = new DataLoader<>(dataset, 5)
                .map(i -> "val_" + i);

        List<String> firstBatch = loader.iterator().next();
        assertEquals(5, firstBatch.size());
        assertEquals("val_0", firstBatch.get(0));
        assertEquals("val_4", firstBatch.get(4));
    }
}
