package tech.kayys.gollek.elasticep;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ElasticExpertParallelismRunnerTest {

    @Test
    void computeRemapsReturnsOnlyChanges() {
        ElasticExpertParallelismRunner runner = new ElasticExpertParallelismRunner();
        int[] prev = new int[]{0, 1, 2, 3};
        int[] next = new int[]{0, 2, 2, 1};

        int[][] remaps = runner.computeRemapsForTest(prev, next);
        assertEquals(2, remaps.length);
        assertArrayEquals(new int[]{1, 1, 2}, remaps[0]);
        assertArrayEquals(new int[]{3, 3, 1}, remaps[1]);
    }

    @Test
    void computeRemapsHandlesNoChanges() {
        ElasticExpertParallelismRunner runner = new ElasticExpertParallelismRunner();
        int[] prev = new int[]{0, 1};
        int[] next = new int[]{0, 1};

        int[][] remaps = runner.computeRemapsForTest(prev, next);
        assertEquals(0, remaps.length);
    }
}
