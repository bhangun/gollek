package tech.kayys.gollek.ml.converter;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.linear_model.LinearModel;
import tech.kayys.gollek.ml.pickle.PickleParser;
import tech.kayys.gollek.ml.tree.DecisionTreeClassifier;
import tech.kayys.gollek.ml.ensemble.GradientBoostingClassifier;
import tech.kayys.gollek.ml.pipeline.PCA;
import tech.kayys.gollek.ml.pipeline.StandardScaler;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class SklearnConverterTest {

    @Test
    public void testConvertLogisticRegression() {
        PickleParser.PickleClass cls = new PickleParser.PickleClass("sklearn.linear_model", "LogisticRegression");
        PickleParser.PickleObject obj = cls.newInstance();
        
        Map<String, Object> state = new HashMap<>();
        state.put("coef_", new double[]{0.1, 0.2, 0.3});
        state.put("intercept_", 0.5);
        state.put("C", 1.0);
        state.put("penalty", "l2");
        obj.setState(state);

        BaseEstimator estimator = SklearnConverter.convert(obj);
        assertTrue(estimator instanceof LinearModel);
        
        LinearModel model = (LinearModel) estimator;
        assertArrayEquals(new double[]{0.1, 0.2, 0.3}, model.getCoefficients());
        assertEquals(0.5, model.getIntercept());
        assertTrue(model.isFitted());
    }

    @Test
    public void testConvertDecisionTree() {
        PickleParser.PickleClass cls = new PickleParser.PickleClass("sklearn.tree", "DecisionTreeClassifier");
        PickleParser.PickleObject obj = cls.newInstance();
        
        Map<String, Object> treeState = new HashMap<>();
        treeState.put("children_left", new int[]{1, -1, -1});
        treeState.put("children_right", new int[]{2, -1, -1});
        treeState.put("feature", new int[]{0, -2, -2});
        treeState.put("threshold", new double[]{0.5, -2, -2});
        treeState.put("value", new double[]{10, 10, 10, 0, 0, 10}); // 2 classes, 3 nodes
        
        Map<String, Object> state = new HashMap<>();
        state.put("tree_", treeState);
        state.put("max_depth", 5);
        obj.setState(state);

        BaseEstimator estimator = SklearnConverter.convert(obj);
        assertTrue(estimator instanceof DecisionTreeClassifier);
        assertTrue(estimator.isFitted());
        
        DecisionTreeClassifier tree = (DecisionTreeClassifier) estimator;
        // Simple prediction check
        assertEquals(0, tree.predictSingle(new float[]{0.1f}));
        assertEquals(1, tree.predictSingle(new float[]{0.9f}));
    }

    @Test
    public void testConvertStandardScaler() {
        PickleParser.PickleClass cls = new PickleParser.PickleClass("sklearn.preprocessing", "StandardScaler");
        PickleParser.PickleObject obj = cls.newInstance();
        
        Map<String, Object> state = new HashMap<>();
        state.put("mean_", new double[]{1.0, 2.0});
        state.put("scale_", new double[]{0.5, 0.5});
        obj.setState(state);

        BaseEstimator estimator = SklearnConverter.convert(obj);
        assertTrue(estimator instanceof StandardScaler);
        assertTrue(estimator.isFitted());
        
        StandardScaler scaler = (StandardScaler) estimator;
        float[] input = {2.0f, 3.0f};
        float[] output = scaler.transformSingle(input);
        assertEquals(2.0f, output[0]); // (2-1)/0.5 = 2
        assertEquals(2.0f, output[1]); // (3-2)/0.5 = 2
    }
}
