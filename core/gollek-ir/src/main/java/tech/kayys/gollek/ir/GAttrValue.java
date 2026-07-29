package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.*;
import tech.kayys.gollek.ir.*;

import tech.kayys.alkhawarizm.core.tensor.*;

import tech.kayys.alkhawarizm.core.tensor.Tensor;

import tech.kayys.gollek.ir.schema.*;
import tech.kayys.gollek.ir.validate.*;

import java.util.*;
import java.nio.file.Path;

/**
 * 
 * 
 * Examples
 * public record GAttrInt(int value) implements GAttrValue {}
 * public record GAttrFloat(float value) implements GAttrValue {}
 * public record GAttrString(String value) implements GAttrValue {}
 * public record GAttrBool(boolean value) implements GAttrValue {}
 * 🔁
 * 
 * This is mandatory for:
 * serialization
 * validation
 * plugin safety
 */
public sealed interface GAttrValue
                permits GAttrInt, GAttrFloat, GAttrString, GAttrBool {
}