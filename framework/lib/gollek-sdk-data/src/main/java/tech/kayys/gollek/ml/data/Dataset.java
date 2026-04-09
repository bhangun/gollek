package tech.kayys.gollek.ml.data;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Base interface for all datasets — equivalent to {@code torch.utils.data.Dataset}.
 */
public interface Dataset<T> {

    int size();

    T get(int index);

    record Sample(GradTensor input, GradTensor label) {}
}
