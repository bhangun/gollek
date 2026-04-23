import jdk.incubator.vector.*;
public class TestFloat {
    public static void main(String[] args) {
        VectorSpecies<Byte> B8 = VectorSpecies.of(byte.class, VectorShape.S_64_BIT);
        VectorSpecies<Float> F8 = VectorSpecies.of(float.class, VectorShape.S_256_BIT);
        byte[] arr = {1, 2, 3, 4, 5, 6, 7, 8};
        ByteVector bvec = ByteVector.fromArray(B8, arr, 0);
        FloatVector fvec = (FloatVector) bvec.castShape(F8, 0);
        float[] out = new float[8];
        fvec.intoArray(out, 0);
        for(float f : out) System.out.println(f);
    }
}
