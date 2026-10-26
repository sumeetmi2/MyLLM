package com.myllm.dao;

/**
 * A function of two floats, used for element-wise matrix operations.
 *
 * <p>The JDK has {@code DoubleBinaryOperator} but no float version.
 */
@FunctionalInterface
public interface FloatOp {
    float apply(float a, float b);
}
