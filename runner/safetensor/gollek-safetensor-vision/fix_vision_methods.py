import re
import os

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Revert duplicate casting issue if any
    content = content.replace('((Map<String, AccelTensor>)(Map<?, ?>)((Map<String, AccelTensor>)(Map<?, ?>)model.weights()))', '((Map<String, AccelTensor>)(Map<?, ?>)model.weights())')

    # Fix .add() and .matmul() ONLY on AccelTensor, but it's hard to distinguish.
    # Let's target the exact instances mentioned in the error.
    # variable patches: patches.matmul(w)
    content = re.sub(r'([a-zA-Z0-9_]+)\.matmul\(([^)]+)\)', r'AccelOps.matmul(\1, \2)', content)

    # For .add(), let's be careful not to replace List.add. 
    # VisionEncoder uses: current.add, h.add, out.add, x.add? Let's explicitly replace:
    content = re.sub(r'(current|h|out|x|visionOut)\.add\(([^)]+)\)', r'AccelOps.add(\1, \2)', content)

    # .softmax(dim) -> AccelOps.softmax(tensor, dim)
    content = re.sub(r'([a-zA-Z0-9_]+)\.softmax\(([^)]+)\)', r'AccelOps.softmax(\1, \2)', content)
    
    with open(path, 'w') as f:
        f.write(content)

base = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/safetensor/gollek-safetensor-vision/src/main/java/tech/kayys/gollek/safetensor/vision'
for file in os.listdir(base):
    if file.endswith('Encoder.java') or file.endswith('Model.java') or file.endswith('Engine.java'):
        patch_file(os.path.join(base, file))

print("Vision methods patched")
