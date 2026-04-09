package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RMSprop optimizer — adapts learning rate per-parameter using a moving
 * average of squared gradients.
 *
 * <p>Update rule:
 * <pre>
 *   v_t = alpha * v_{t-1} + (1 - alpha) * g^2
 *   theta -= lr / (sqrt(v_t) + eps) * g
 * </pre>
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for the parameter update loop.
 */
public class RMSprop implements Optimizer {

     private final List<Parameter> parameters;
     private float lr;
     private final float alpha;   // smoothing constant (default 0.99)
     private final float eps;
     private final float weightDecay;
     private final Map<Parameter, float[]> squaredAvg = new HashMap<>();

     public RMSprop(List<Parameter> params, float lr) {
         this(params, lr, 0.99f, 1e-8f, 0f);
     }

     public RMSprop(List<Parameter> params, float lr, float alpha, float eps, float weightDecay) {
         this.parameters = params;
         this.lr = lr;
         this.alpha = alpha;
         this.eps = eps;
         this.weightDecay = weightDecay;
     }

     @Override
     public void step() {
         for (Parameter p : parameters) {
             if (p.data().grad() == null) continue;
             float[] data = p.data().data();
             float[] grad = p.data().grad().data();
             float[] v    = squaredAvg.computeIfAbsent(p, k -> new float[data.length]);

             int len = data.length;
             int bound = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.loopBound(len);
             var SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
             var alphaV  = jdk.incubator.vector.FloatVector.broadcast(SPECIES, alpha);
             var oneMinusAlpha = jdk.incubator.vector.FloatVector.broadcast(SPECIES, 1f - alpha);
             var lrV     = jdk.incubator.vector.FloatVector.broadcast(SPECIES, lr);
             var epsV    = jdk.incubator.vector.FloatVector.broadcast(SPECIES, eps);

             int i = 0;
             for (; i < bound; i += SPECIES.length()) {
                 var g  = jdk.incubator.vector.FloatVector.fromArray(SPECIES, grad, i);
                 var vi = jdk.incubator.vector.FloatVector.fromArray(SPECIES, v, i);
                 // v = alpha*v + (1-alpha)*g^2
                 var newV = alphaV.mul(vi).add(oneMinusAlpha.mul(g.mul(g)));
                 newV.intoArray(v, i);
                 // theta -= lr / (sqrt(v) + eps) * g
                 var denom = newV.sqrt().add(epsV);
                 var d = jdk.incubator.vector.FloatVector.fromArray(SPECIES, data, i);
                 d.sub(lrV.mul(g).div(denom)).intoArray(data, i);
             }
             for (; i < len; i++) {
                 float g = grad[i];
                 if (weightDecay != 0f) g += weightDecay * data[i];
                 v[i] = alpha * v[i] + (1f - alpha) * g * g;
                 data[i] -= lr / ((float) Math.sqrt(v[i]) + eps) * g;
             }
         }
     }

     @Override
     public void zeroGrad() {
         parameters.forEach(p -> p.zeroGrad());
     }

     @Override
     public float learningRate() {
         return lr;
     }

     @Override
     public List<Parameter> parameters() {
         return parameters;
     }

     @Override
     public void setLearningRate(float newLr) {
         this.lr = newLr;
     }
}
