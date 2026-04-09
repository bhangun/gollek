package tech.kayys.gollek.ml.nn.loss;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * IoU Loss (Jaccard Loss) — 1 - IoU, for object detection bounding box regression.
 *
 * <p>Directly optimizes the IoU metric rather than a surrogate loss.
 * More robust to scale than MSE/SmoothL1 for bounding boxes.
 *
 * <p>Expects boxes in {@code [x1, y1, x2, y2]} format.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var loss = new IoULoss();
 * // pred/target: [N, 4] in (x1,y1,x2,y2) format
 * GradTensor l = loss.forward(predBoxes, targetBoxes);
 * }</pre>
 */
public final class IoULoss {

    /**
     * Computes mean IoU loss over a batch of bounding boxes.
     *
     * @param pred   predicted boxes {@code [N, 4]} in (x1,y1,x2,y2) format
     * @param target ground-truth boxes {@code [N, 4]}
     * @return scalar mean IoU loss (1 - mean_IoU)
     */
    public GradTensor forward(GradTensor pred, GradTensor target) {
        int N = (int) pred.shape()[0];
        float[] p = pred.data(), t = target.data();
        float[] ious = new float[N];

        for (int n = 0; n < N; n++) {
            float px1 = p[n*4], py1 = p[n*4+1], px2 = p[n*4+2], py2 = p[n*4+3];
            float tx1 = t[n*4], ty1 = t[n*4+1], tx2 = t[n*4+2], ty2 = t[n*4+3];

            float interX1 = Math.max(px1, tx1), interY1 = Math.max(py1, ty1);
            float interX2 = Math.min(px2, tx2), interY2 = Math.min(py2, ty2);
            float interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);

            float predArea   = Math.max(0, px2 - px1) * Math.max(0, py2 - py1);
            float targetArea = Math.max(0, tx2 - tx1) * Math.max(0, ty2 - ty1);
            float unionArea  = predArea + targetArea - interArea + 1e-7f;

            ious[n] = interArea / unionArea;
        }
        return GradTensor.scalar(1f - VectorOps.sum(ious) / N);
    }
}
