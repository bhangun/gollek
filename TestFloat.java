public class TestFloat {
    public static void main(String[] args) {
        float[] v = { 10.0f, 20.0f, 30.0f };
        float temp = 0.0f;
        int best = -1;
        float max = Float.NEGATIVE_INFINITY;
        for(int i=0;i<v.length;i++) {
            float scaled = v[i] / temp;
            if (scaled > max) { max = scaled; best = i; }
        }
        System.out.println("Best: " + best + ", Max: " + max);
    }
}
