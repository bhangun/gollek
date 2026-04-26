package tech.kayys.gollek.ml.serialize;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import tech.kayys.gollek.ml.base.BaseEstimator;

/**
 * Python pickle compatible serialization for Gollek models.
 * Implements pickle protocol version 4 (PEP 3154).
 */
public class PickleSerializer {

    // Pickle opcodes
    private static final byte PROTO = (byte) 0x80;
    private static final byte FRAME = (byte) 0x95;
    private static final byte EMPTY_TUPLE = (byte) 0x29;
    private static final byte EMPTY_LIST = (byte) 0x5D;
    private static final byte EMPTY_DICT = (byte) 0x7D;
    private static final byte MARK = (byte) 0x28;
    private static final byte STOP = (byte) 0x2E;
    private static final byte BINUNICODE = (byte) 0x88;
    private static final byte BININT = (byte) 0x4A;
    private static final byte BININT1 = (byte) 0x4B;
    private static final byte BININT2 = (byte) 0x4C;
    private static final byte BINFLOAT = (byte) 0x47;
    private static final byte BINBYTES = (byte) 0x42;
    private static final byte SHORT_BINBYTES = (byte) 0x43;
    private static final byte BINUNICODE8 = (byte) 0x8D;
    private static final byte GLOBAL = (byte) 0x63;
    private static final byte NEWOBJ = (byte) 0x85;
    private static final byte BUILD = (byte) 0x86;
    private static final byte SETITEM = (byte) 0x94;
    private static final byte TUPLE = (byte) 0x28;
    private static final byte LIST = (byte) 0x5D;
    private static final byte DICT = (byte) 0x7D;
    private static final byte APPENDS = (byte) 0x65;
    private static final byte SETITEMS = (byte) 0x75;

    /**
     * Save model to pickle file (Python-compatible).
     */
    public static void toPickle(BaseEstimator model, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path)))) {

            // Write pickle protocol header
            out.writeByte(PROTO);
            out.writeByte(4); // Protocol version 4

            // Write the model
            writePickle(out, model);

            // Write STOP opcode
            out.writeByte(STOP);
            out.flush();
        }
    }

    /**
     * Save model with compression.
     */
    public static void toPickleGZ(BaseEstimator model, String path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new GZIPOutputStream(new FileOutputStream(path)))) {
            out.writeByte(PROTO);
            out.writeByte(4);
            writePickle(out, model);
            out.writeByte(STOP);
            out.flush();
        }
    }

    /**
     * Load model from pickle file.
     */
    @SuppressWarnings("unchecked")
    public static <T extends BaseEstimator> T fromPickle(String path) throws IOException, ClassNotFoundException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path)))) {

            // Read and verify protocol
            byte proto = in.readByte();
            if (proto != PROTO) {
                throw new IOException("Invalid pickle file: missing PROTO");
            }
            byte version = in.readByte();
            if (version < 4) {
                throw new IOException("Unsupported pickle protocol version: " + version);
            }

            return (T) readPickle(in);
        }
    }

    /**
     * Write any object in pickle format.
     */
    private static void writePickle(DataOutputStream out, Object obj) throws IOException {
        if (obj == null) {
            writeNone(out);
        } else if (obj instanceof BaseEstimator) {
            writeEstimator(out, (BaseEstimator) obj);
        } else if (obj instanceof float[][]) {
            writeFloatMatrix(out, (float[][]) obj);
        } else if (obj instanceof float[]) {
            writeFloatArray(out, (float[]) obj);
        } else if (obj instanceof int[][]) {
            writeIntMatrix(out, (int[][]) obj);
        } else if (obj instanceof int[]) {
            writeIntArray(out, (int[]) obj);
        } else if (obj instanceof double[][]) {
            writeDoubleMatrix(out, (double[][]) obj);
        } else if (obj instanceof double[]) {
            writeDoubleArray(out, (double[]) obj);
        } else if (obj instanceof Integer) {
            writeInt(out, (Integer) obj);
        } else if (obj instanceof Float) {
            writeFloat(out, (Float) obj);
        } else if (obj instanceof Double) {
            writeDouble(out, (Double) obj);
        } else if (obj instanceof Boolean) {
            writeBool(out, (Boolean) obj);
        } else if (obj instanceof String) {
            writeString(out, (String) obj);
        } else if (obj instanceof List) {
            writeList(out, (List<?>) obj);
        } else if (obj instanceof Map) {
            writeDict(out, (Map<?, ?>) obj);
        } else if (obj instanceof Object[]) {
            writeArray(out, (Object[]) obj);
        } else {
            throw new IOException("Unsupported pickle type: " + obj.getClass().getName());
        }
    }

    /**
     * Read object from pickle format.
     */
    private static Object readPickle(DataInputStream in) throws IOException {
        int opcode = in.readByte() & 0xFF;

        switch (opcode) {
            case BININT:
                return in.readInt();
            case BININT1:
                return (int) in.readByte();
            case BININT2:
                return (int) in.readShort();
            case BINFLOAT:
                return in.readDouble();
            case BINUNICODE:
                return readString(in);
            case BINUNICODE8:
                return readString8(in);
            case SHORT_BINBYTES:
                return readBytes(in);
            case EMPTY_TUPLE:
                return new Object[0];
            case EMPTY_LIST:
                return new ArrayList<>();
            case EMPTY_DICT:
                return new LinkedHashMap<>();
            case NONE:
                return null;
            case GLOBAL:
                return readGlobal(in);
            case NEWOBJ:
                return readNewObj(in);
            case BUILD:
                return readBuild(in);
            default:
                throw new IOException("Unsupported pickle opcode: " + opcode);
        }
    }

    /**
     * Write None/null value.
     */
    private static void writeNone(DataOutputStream out) throws IOException {
        out.writeByte(0x4E); // 'N' for None
    }

    private static final int NONE = 0x4E;

    /**
     * Write integer with minimal encoding.
     */
    private static void writeInt(DataOutputStream out, int value) throws IOException {
        if (value >= -128 && value <= 127) {
            out.writeByte(BININT1);
            out.writeByte(value);
        } else if (value >= -32768 && value <= 32767) {
            out.writeByte(BININT2);
            out.writeShort(value);
        } else {
            out.writeByte(BININT);
            out.writeInt(value);
        }
    }

    /**
     * Write float.
     */
    private static void writeFloat(DataOutputStream out, float value) throws IOException {
        out.writeByte(BINFLOAT);
        out.writeDouble(value);
    }

    /**
     * Write double.
     */
    private static void writeDouble(DataOutputStream out, double value) throws IOException {
        out.writeByte(BINFLOAT);
        out.writeDouble(value);
    }

    /**
     * Write boolean.
     */
    private static void writeBool(DataOutputStream out, boolean value) throws IOException {
        out.writeByte(0x88); // NEWTRUE or NEWFALSE
        out.writeBoolean(value);
    }

    /**
     * Write string.
     */
    private static void writeString(DataOutputStream out, String value) throws IOException {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        if (bytes.length < 256) {
            out.writeByte(SHORT_BINBYTES);
            out.writeByte(bytes.length);
        } else if (bytes.length < 65536) {
            out.writeByte(BINBYTES);
            out.writeInt(bytes.length);
        } else {
            out.writeByte(BINUNICODE8);
            out.writeLong(bytes.length);
        }
        out.write(bytes);
    }

    /**
     * Write float array.
     */
    private static void writeFloatArray(DataOutputStream out, float[] array) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (float v : array) {
            writeFloat(out, v);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write float matrix (2D array).
     */
    private static void writeFloatMatrix(DataOutputStream out, float[][] matrix) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (float[] row : matrix) {
            writeFloatArray(out, row);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write int array.
     */
    private static void writeIntArray(DataOutputStream out, int[] array) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (int v : array) {
            writeInt(out, v);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write int matrix.
     */
    private static void writeIntMatrix(DataOutputStream out, int[][] matrix) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (int[] row : matrix) {
            writeIntArray(out, row);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write double array.
     */
    private static void writeDoubleArray(DataOutputStream out, double[] array) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (double v : array) {
            writeDouble(out, v);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write double matrix.
     */
    private static void writeDoubleMatrix(DataOutputStream out, double[][] matrix) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (double[] row : matrix) {
            writeDoubleArray(out, row);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write list.
     */
    private static void writeList(DataOutputStream out, List<?> list) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (Object item : list) {
            writePickle(out, item);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write dictionary/map.
     */
    private static void writeDict(DataOutputStream out, Map<?, ?> map) throws IOException {
        out.writeByte(EMPTY_DICT);
        out.writeByte(MARK);
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            writePickle(out, entry.getKey());
            writePickle(out, entry.getValue());
        }
        out.writeByte(SETITEMS);
    }

    /**
     * Write array.
     */
    private static void writeArray(DataOutputStream out, Object[] array) throws IOException {
        out.writeByte(EMPTY_LIST);
        out.writeByte(MARK);
        for (Object item : array) {
            writePickle(out, item);
        }
        out.writeByte(APPENDS);
    }

    /**
     * Write estimator (custom object).
     */
    private static void writeEstimator(DataOutputStream out, BaseEstimator estimator) throws IOException {
        // Write global reference (module and class name)
        String moduleName = estimator.getClass().getPackage().getName();
        String className = estimator.getClass().getSimpleName();

        out.writeByte(GLOBAL);
        writeString(out, moduleName);
        writeString(out, className);

        // Write object state as dictionary
        Map<String, Object> state = extractState(estimator);
        out.writeByte(EMPTY_TUPLE); // Args tuple (empty for __new__)
        out.writeByte(NEWOBJ);

        // Write state as dict for __setstate__
        writeDict(out, state);
        out.writeByte(BUILD);
    }

    /**
     * Read string from pickle.
     */
    private static String readString(DataInputStream in) throws IOException {
        int len = in.readInt();
        byte[] bytes = new byte[len];
        in.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Read long string from pickle.
     */
    private static String readString8(DataInputStream in) throws IOException {
        long len = in.readLong();
        if (len > Integer.MAX_VALUE) {
            throw new IOException("String too long: " + len);
        }
        byte[] bytes = new byte[(int) len];
        in.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Read bytes from pickle.
     */
    private static byte[] readBytes(DataInputStream in) throws IOException {
        int len = in.readUnsignedByte();
        byte[] bytes = new byte[len];
        in.readFully(bytes);
        return bytes;
    }

    /**
     * Read global reference.
     */
    private static Object readGlobal(DataInputStream in) throws IOException {
        String module = readString(in);
        String name = readString(in);
        // Return a placeholder - actual object instantiation happens later
        return new GlobalReference(module, name);
    }

    /**
     * Read new object creation.
     */
    private static Object readNewObj(DataInputStream in) throws IOException {
        // Skip the tuple (already read)
        return new Object();
    }

    /**
     * Read build/state restoration.
     */
    private static Object readBuild(DataInputStream in) throws IOException {
        // Read the state dictionary
        @SuppressWarnings("unchecked")
        Map<String, Object> state = (Map<String, Object>) readPickle(in);
        return state;
    }

    /**
     * Extract estimator state for pickling.
     */
    private static Map<String, Object> extractState(BaseEstimator estimator) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("_class", estimator.getClass().getName());
        state.put("_module", estimator.getClass().getPackage().getName());

        // Use reflection to get all fields
        try {
            for (java.lang.reflect.Field field : estimator.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object value = field.get(estimator);
                if (value != null && isPicklable(value)) {
                    state.put(field.getName(), value);
                }
            }
        } catch (IllegalAccessException e) {
            throw new RuntimeException("Failed to extract estimator state", e);
        }

        return state;
    }

    /**
     * Check if object is picklable.
     */
    private static boolean isPicklable(Object obj) {
        return obj instanceof Number ||
                obj instanceof String ||
                obj instanceof Boolean ||
                obj instanceof float[] ||
                obj instanceof int[] ||
                obj instanceof double[] ||
                obj instanceof float[][] ||
                obj instanceof int[][] ||
                obj instanceof double[][] ||
                obj instanceof List ||
                obj instanceof Map ||
                obj.getClass().isArray();
    }

    /**
     * Placeholder for global references.
     */
    static class GlobalReference {
        String module;
        String name;

        GlobalReference(String module, String name) {
            this.module = module;
            this.name = name;
        }
    }
}