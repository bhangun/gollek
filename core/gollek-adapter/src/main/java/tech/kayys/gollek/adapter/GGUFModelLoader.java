package tech.kayys.gollek.adapter;

import tech.kayys.gollek.core.graph.*;
import tech.kayys.gollek.core.graph.node.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.alkhawarizm.core.tensor.ModelWeightLoader;
import java.nio.file.Path;

public final class GGUFModelLoader implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".gguf");
    }

    @Override
    public WeightAdapter load(Path path) {
        return new GGUFAdapter();
    }
}
