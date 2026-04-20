import re
import os

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Import
    content = content.replace('import tech.kayys.gollek.inference.libtorch.core.TorchTensor;', 
                              'import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;\nimport tech.kayys.gollek.safetensor.core.tensor.AccelOps;')
    
    # Types
    content = content.replace('TorchTensor', 'AccelTensor')

    # Fix Map casts
    content = content.replace('Map<String, AccelTensor> weights = (Map<String, AccelTensor>) model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> weights = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')
    content = content.replace('Map<String, AccelTensor> w = (Map<String, AccelTensor>) model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> w = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')
    content = content.replace('Map<String, AccelTensor> weights = model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> weights = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')
    content = content.replace('Map<String, AccelTensor> w = model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> w = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')

    # .add() / etc shouldn't be blindly replaced by AccelOps.add. Let's see if the vision engine did any math. 
    # Usually Vision just delegates to CLIP / ViT.
    # Wait, the error is incompatible types, maybe there are only casting errors.
    
    with open(path, 'w') as f:
        f.write(content)

base = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/safetensor/gollek-safetensor-vision/src/main/java/tech/kayys/gollek/safetensor/vision'
patch_file(os.path.join(base, 'MultimodalInferenceEngine.java'))

print("Vision patched")
