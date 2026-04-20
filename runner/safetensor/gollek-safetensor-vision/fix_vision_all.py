import re
import os

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Sometimes people use Map<String, TorchTensor> inline or in method signatures
    content = content.replace('Map<String, TorchTensor>', 'Map<String, AccelTensor>')

    # Convert model.weights() safely
    content = content.replace('model.weights()', '((Map<String, AccelTensor>)(Map<?, ?>)model.weights())')

    # Revert duplicate casting issue if any created by previous run
    content = content.replace('((Map<String, AccelTensor>)(Map<?, ?>)((Map<String, AccelTensor>)(Map<?, ?>)model.weights()))', '((Map<String, AccelTensor>)(Map<?, ?>)model.weights())')

    # Remove any remaining libtorch import traces which could cause weird type errors
    content = content.replace('tech.kayys.gollek.inference.libtorch.core.TorchTensor', 'tech.kayys.gollek.safetensor.core.tensor.AccelTensor')
    content = content.replace('import tech.kayys.gollek.inference.libtorch.core.TorchTensor;', 
                              'import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;\nimport tech.kayys.gollek.safetensor.core.tensor.AccelOps;')
    content = content.replace('TorchTensor', 'AccelTensor')


    with open(path, 'w') as f:
        f.write(content)

base = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/safetensor/gollek-safetensor-vision/src/main/java/tech/kayys/gollek/safetensor/vision'
for file in os.listdir(base):
    if file.endswith('.java'):
        patch_file(os.path.join(base, file))

print("Vision patched thoroughly")
