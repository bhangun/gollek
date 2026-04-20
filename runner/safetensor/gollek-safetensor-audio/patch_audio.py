import re
import os

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Import
    content = content.replace('import tech.kayys.gollek.inference.libtorch.core.TorchTensor;', 
                              'import tech.kayys.gollek.safetensor.engine.tensor.AccelTensor;\nimport tech.kayys.gollek.safetensor.engine.tensor.AccelOps;')
    
    # Types
    content = content.replace('TorchTensor', 'AccelTensor')

    # .add()
    content = re.sub(r'([a-zA-Z0-9_]+)\.add\(([a-zA-Z0-9_]+)\)', r'AccelOps.add(\1, \2)', content)

    # linear implementation
    content = content.replace('x.matmul(w.transpose(w.shape().length - 2, w.shape().length - 1))', 'AccelOps.linear(x, w)')
    
    # matmul
    content = re.sub(r'([a-zA-Z0-9_]+)\.matmul\(([^)]+)\)', r'AccelOps.matmul(\1, \2)', content)

    # selfAttention and crossAttention softmax logic
    #   AccelTensor scores = AccelOps.matmul(q, encOut.transpose(1, 2))
    #           .mul(AccelTensor.fromFloatArray(new float[] { scale }, new long[] { 1 }));
    content = content.replace('AccelTensor exp = scores.exp();\n            AccelTensor sum = exp.sum();\n            AccelTensor attnW = exp.div(sum);',
                              'AccelTensor attnW = AccelOps.softmax(scores, -1);')
    content = content.replace('AccelTensor exp = sc.exp();\n        AccelTensor sum = exp.sum();\n        AccelTensor aw = exp.div(sum);',
                              'AccelTensor aw = AccelOps.softmax(sc, -1);')
    
    # .mul(AccelTensor.fromFloatArray...) -> AccelOps.mulScalar(...)
    content = re.sub(r'([a-zA-Z0-9_]+)\.mul\([^\)]+new float\[\]\s*\{\s*scale\s*\}[^\)]+\)', r'AccelOps.mulScalar(\1, scale)', content)

    # Map<?, ?> w = model.weights(); -> Map<String, AccelTensor> w = (Map<String, AccelTensor>) model.weights();
    content = content.replace('Map<String, AccelTensor> w = model.weights();', 'Map<String, AccelTensor> w = (Map<String, AccelTensor>) model.weights();')
    content = content.replace('Map<String, AccelTensor> weights = model.weights();', 'Map<String, AccelTensor> weights = (Map<String, AccelTensor>) model.weights();')

    # .mul() generic
    content = re.sub(r'([a-zA-Z0-9_]+)\.mul\(([a-zA-Z0-9_]+)\)', r'AccelOps.mul(\1, \2)', content)

    # .div() generic
    content = re.sub(r'([a-zA-Z0-9_]+)\.div\(([a-zA-Z0-9_]+)\)', r'AccelOps.div(\1, \2)', content)

    with open(path, 'w') as f:
        f.write(content)

base = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/safetensor/gollek-safetensor-audio/src/main/java/tech/kayys/gollek/safetensor/audio'
patch_file(os.path.join(base, 'SpeechT5Engine.java'))
patch_file(os.path.join(base, 'WhisperEngine.java'))

print("Source patched")
