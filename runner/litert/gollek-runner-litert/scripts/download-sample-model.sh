#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="${GOLLEK_TFLITE_MODEL_URL:-https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.litertlm}"
OUT_DIR="${GOLLEK_TFLITE_MODEL_DIR:-$HOME/.gollek/models/litert}"

mkdir -p "${OUT_DIR}"
FILE_NAME="${MODEL_URL##*/}"
OUT_PATH="${OUT_DIR}/${FILE_NAME}"

echo "Downloading TFLite sample model..."
echo "URL: ${MODEL_URL}"
echo "OUT: ${OUT_PATH}"

curl -L --fail --retry 3 --retry-delay 2 -o "${OUT_PATH}" "${MODEL_URL}"

echo "✓ Saved ${OUT_PATH}"
