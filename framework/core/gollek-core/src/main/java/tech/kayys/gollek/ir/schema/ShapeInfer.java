package tech.kayys.gollek.ir.schema;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.gollek.ir.*;
import java.util.*;

public interface ShapeInfer {
    Shape[] infer(
            List<GValue> inputs,
            Map<String, GAttrValue> attrs);
}