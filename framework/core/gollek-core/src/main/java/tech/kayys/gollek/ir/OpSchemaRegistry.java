package tech.kayys.gollek.ir;

import java.util.*;

public final class OpSchemaRegistry {
    private final Map<String, OpSchema> schemas = new HashMap<>();

    public void register(OpSchema schema) {
        schemas.put(schema.opType(), schema);
    }

    public OpSchema get(String opType) {
        return schemas.get(opType);
    }
}