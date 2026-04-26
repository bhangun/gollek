package tech.kayys.gollek.cli;

import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;

@RegisterForReflection(targets = {
    ModelConfig.class,
    ModelConfig.RopeScaling.class,
    SafetensorHeader.class,
    SafetensorTensorInfo.class
})
public class NativeRegistrations {
}
