package tech.kayys.gollek.ml.nn.metrics;

import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Intersection over Union (IoU) and Dice Score for segmentation evaluation.
 *
 * <p>Both metrics measure overlap between predicted and ground-truth masks.
 * IoU is the standard metric for object detection (COCO, Pascal VOC).
 *
 * <h3>Formulas</h3>
 * <pre>
 *   IoU  = |A ∩ B| / |A ∪ B|  =  TP / (TP + FP + FN)
 *   Dice = 2|A ∩ B| / (|A| + |B|)  =  2·TP / (2·TP + FP + FN)
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * float iou  = SegmentationMetrics.iou(predicted, target);
 * float dice = SegmentationMetrics.dice(predicted, target);
 * }</pre>
 */
public final class SegmentationMetrics {

    private SegmentationMetrics() {}

    /**
     * Computes the Intersection over Union (IoU / Jaccard index).
     *
     * @param pred   predicted binary mask (0 or 1 values), flat array
     * @param target ground-truth binary mask, flat array
     * @return IoU in [0, 1]
     */
    public static float iou(float[] pred, float[] target) {
        float[] inter = new float[pred.length], union = new float[pred.length];
        for (int i = 0; i < pred.length; i++) {
            inter[i] = Math.min(pred[i], target[i]);
            union[i] = Math.max(pred[i], target[i]);
        }
        float interSum = VectorOps.sum(inter);
        float unionSum = VectorOps.sum(union);
        return unionSum > 0 ? interSum / unionSum : 1f;
    }

    /**
     * Computes the Dice coefficient (F1 score over pixels).
     *
     * @param pred   predicted binary mask, flat array
     * @param target ground-truth binary mask, flat array
     * @return Dice score in [0, 1]
     */
    public static float dice(float[] pred, float[] target) {
        float[] inter = new float[pred.length];
        for (int i = 0; i < pred.length; i++) inter[i] = pred[i] * target[i];
        float interSum = VectorOps.sum(inter);
        float sumP = VectorOps.sum(pred), sumT = VectorOps.sum(target);
        return (sumP + sumT) > 0 ? 2f * interSum / (sumP + sumT) : 1f;
    }

    /**
     * Computes mean IoU over multiple classes.
     *
     * @param pred       predicted class mask {@code [N]} (integer class indices as float)
     * @param target     ground-truth class mask {@code [N]}
     * @param numClasses number of classes
     * @return mean IoU across all classes
     */
    public static float meanIoU(float[] pred, float[] target, int numClasses) {
        float total = 0f;
        int valid = 0;
        for (int c = 0; c < numClasses; c++) {
            float[] pBin = new float[pred.length], tBin = new float[target.length];
            for (int i = 0; i < pred.length; i++) {
                pBin[i] = pred[i] == c ? 1f : 0f;
                tBin[i] = target[i] == c ? 1f : 0f;
            }
            // Skip classes not present in ground truth
            if (VectorOps.sum(tBin) == 0) continue;
            total += iou(pBin, tBin);
            valid++;
        }
        return valid > 0 ? total / valid : 0f;
    }
}
