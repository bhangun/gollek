package tech.kayys.gollek.cache;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

public class PagedKVCacheCoreTest {

    @Test
    public void basicPutGetViewAndPinUnpin() {
        PagedKVCache cache = new PagedKVCache(1024 * 1024);
        long key = 1L;
        float[] data = new float[128];
        for (int i = 0; i < data.length; i++) data[i] = i;

        cache.put(key, data);
        ByteBuffer view = cache.getView(key);
        Assertions.assertNotNull(view);
        Assertions.assertEquals(data.length * Float.BYTES, view.capacity());

        ByteBuffer pinned = cache.pin(key);
        Assertions.assertNotNull(pinned);
        cache.unpin(key);
    }
}
