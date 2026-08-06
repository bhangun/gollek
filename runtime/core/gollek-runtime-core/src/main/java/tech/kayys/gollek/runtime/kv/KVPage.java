package tech.kayys.gollek.runtime.kv;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

public final class KVPage {
    public Tensor key;
    public Tensor value;
    public boolean used;

    public KVPage(Tensor key, Tensor value) {
        this.key = key;
        this.value = value;
        this.used = false;
    }
}