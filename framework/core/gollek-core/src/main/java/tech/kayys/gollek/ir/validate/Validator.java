package tech.kayys.gollek.ir.validate;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.gollek.ir.*;
import java.util.*;

public interface Validator {
    void validate(List<GValue> inputs, Map<String, GAttrValue> attrs);
}