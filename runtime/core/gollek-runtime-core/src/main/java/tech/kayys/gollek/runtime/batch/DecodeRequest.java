package tech.kayys.gollek.runtime.batch;

import tech.kayys.gollek.runtime.kv.KVCache;
import tech.kayys.gollek.runtime.stream.TokenStreamer;

/**
 * CORE IDEA
 * Instead of:
 * 1 request → full decode → next request ❌
 * We do:
 * N requests → interleaved decode → shared compute ✅
 */
public final class DecodeRequest {
    public final String id;
    public final KVCache kv;
    public final TokenStreamer streamer;
    public int position = 0;
    public boolean finished = false;

    public DecodeRequest(String id,
            KVCache kv,
            TokenStreamer streamer) {
        this.id = id;
        this.kv = kv;
        this.streamer = streamer;
    }
}