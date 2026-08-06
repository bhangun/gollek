package tech.kayys.gollek.core.data;


import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

public final class GHandle implements GData {
    private final String id;
    private final Object ref;

    public GHandle(String id, Object ref) {
        this.id = id;
        this.ref = ref;
    }

    public Object ref() {
        return ref;
    }

    @Override
    public String id() {
        return id;
    }
}
