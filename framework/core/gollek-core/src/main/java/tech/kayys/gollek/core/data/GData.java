package tech.kayys.gollek.core.data;


import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;
import java.util.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

/**
 * Unified data contract for all Gollek runtime data types.
 */
public sealed interface GData permits GTensor, GScalar, GHandle {
    String id();
}
