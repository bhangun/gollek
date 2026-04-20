import re
import os

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Revert bad AccelOps.add lists
    content = re.sub(r'AccelOps\.add\((melFrames|tokens|chunks|segments|melFrames),\s*([^)]+)\)', r'\1.add(\2)', content)

    # Fix Map casts
    content = content.replace('Map<String, AccelTensor> w = (Map<String, AccelTensor>) model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> w = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')
    content = content.replace('Map<String, AccelTensor> weights = (Map<String, AccelTensor>) model.weights();', 
                              '@SuppressWarnings("unchecked") Map<String, AccelTensor> weights = (Map<String, AccelTensor>) (Map<?, ?>) model.weights();')
                              
    # Fix .mul(AccelTensor) with mulScalar
    content = re.sub(
        r'AccelTensor scores = AccelOps\.matmul\(([^,]+),\s*encOut\.transpose\(1, 2\)\)\s*\.mul\(AccelTensor\.fromFloatArray\(new float\[\] \{ scale \}, new long\[\] \{ 1 \}\)\);',
        r'AccelTensor scoresTemp = AccelOps.matmul(\1, encOut.transpose(1, 2));\n            AccelTensor scores = AccelOps.mulScalar(scoresTemp, scale);\n            scoresTemp.close();',
        content
    )
    content = re.sub(
        r'AccelTensor sc = AccelOps\.matmul\(([^,]+),\s*k\.transpose\(1, 2\)\)\.mul\(AccelTensor\.fromFloatArray\(new float\[\] \{ scale \}, new long\[\] \{ 1 \}\)\);',
        r'AccelTensor scTemp = AccelOps.matmul(\1, k.transpose(1, 2));\n        AccelTensor sc = AccelOps.mulScalar(scTemp, scale);\n        scTemp.close();',
        content
    )

    # Note: AccelTensor doesn't have .mul() anymore since we're porting directly.
    # We replaced most x.mul() with AccelOps.mul() in previous step? Wait, no, we only replaced `.add()`.
    # Actually wait. Let's look at `encoderOut.transpose(1, 2)) .mul(AccelTensor...` it failed with "method mul(AccelTensor) not found"
    
    # Remove exp.close() and sum.close()
    content = content.replace('exp.close();\n            sum.close();\n', '')
    
    with open(path, 'w') as f:
        f.write(content)

base = '/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/safetensor/gollek-safetensor-audio/src/main/java/tech/kayys/gollek/safetensor/audio'
patch_file(os.path.join(base, 'SpeechT5Engine.java'))
patch_file(os.path.join(base, 'WhisperEngine.java'))

print("Source patched")
