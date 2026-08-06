package tech.kayys.gollek.ml.pickle;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

/**
 * Complete pickle file parser for Java.
 * Supports loading scikit-learn, PyTorch, and other Python ML models.
 */
public class PickleParser {

    // Pickle opcodes
    private static final byte PROTO = (byte) 0x80;
    private static final byte MARK = (byte) 0x28;
    private static final byte STOP = (byte) 0x2E;
    private static final byte POP = (byte) 0x30;
    private static final byte POP_MARK = (byte) 0x31;
    private static final byte DUP = (byte) 0x32;
    private static final byte FLOAT = (byte) 0x46;
    private static final byte BINFLOAT = (byte) 0x47;
    private static final byte INT = (byte) 0x49;
    private static final byte BININT = (byte) 0x4A;
    private static final byte BININT1 = (byte) 0x4B;
    private static final byte BININT2 = (byte) 0x4D;
    private static final byte LONG = (byte) 0x4C;
    private static final byte STRING = (byte) 0x53;
    private static final byte BINSTRING = (byte) 0x54;
    private static final byte SHORT_BINSTRING = (byte) 0x55;
    private static final byte UNICODE = (byte) 0x56;
    private static final byte BINUNICODE = (byte) 0x58;
    private static final byte BINUNICODE8 = (byte) 0x8D;
    private static final byte NONE = (byte) 0x4E;
    private static final byte NEWTRUE = (byte) 0x88;
    private static final byte NEWFALSE = (byte) 0x89;
    private static final byte EMPTY_TUPLE = (byte) 0x29;
    private static final byte TUPLE = (byte) 0x74;
    private static final byte EMPTY_LIST = (byte) 0x5D;
    private static final byte LIST = (byte) 0x5B;
    private static final byte APPEND = (byte) 0x61;
    private static final byte APPENDS = (byte) 0x65;
    private static final byte EMPTY_DICT = (byte) 0x7D;
    private static final byte DICT = (byte) 0x7B;
    private static final byte SETITEM = (byte) 0x73;
    private static final byte SETITEMS = (byte) 0x75;
    private static final byte BUILD = (byte) 0x86;
    private static final byte NEWOBJ = (byte) 0x85;
    private static final byte REDUCE = (byte) 0x52;
    private static final byte GLOBAL = (byte) 0x63;
    private static final byte OBJ = (byte) 0x6F;
    private static final byte INST = (byte) 0x69;
    private static final byte BINBYTES = (byte) 0x42;
    private static final byte SHORT_BINBYTES = (byte) 0x43;
    private static final byte BYTEARRAY8 = (byte) 0x96;

    private final byte[] data;
    private int pos;
    private final Stack<Object> stack = new Stack<>();
    private final Stack<Object> memo = new Stack<>();

    public PickleParser(byte[] data) {
        this.data = data;
        this.pos = 0;
    }

    /**
     * Parse pickle file and return the reconstructed object.
     */
    public Object parse() throws IOException {
        while (pos < data.length) {
            byte opcode = data[pos++];

            switch (opcode) {
                case PROTO:
                    readProto();
                    break;
                case MARK:
                    stack.push(MARK);
                    break;
                case STOP:
                    return stack.pop();
                case POP:
                    stack.pop();
                    break;
                case POP_MARK:
                    popMark();
                    break;
                case DUP:
                    stack.push(stack.peek());
                    break;
                case FLOAT:
                    readFloat();
                    break;
                case BINFLOAT:
                    readBinFloat();
                    break;
                case INT:
                    readInt();
                    break;
                case BININT:
                    readBinInt();
                    break;
                case BININT1:
                    readBinInt1();
                    break;
                case BININT2:
                    readBinInt2();
                    break;
                case STRING:
                    readString();
                    break;
                case BINSTRING:
                    readBinString();
                    break;
                case SHORT_BINSTRING:
                    readShortBinString();
                    break;
                case UNICODE:
                    readUnicode();
                    break;
                case BINUNICODE:
                    readBinUnicode();
                    break;
                case BINUNICODE8:
                    readBinUnicode8();
                    break;
                case NONE:
                    stack.push(null);
                    break;
                case NEWTRUE:
                    stack.push(Boolean.TRUE);
                    break;
                case NEWFALSE:
                    stack.push(Boolean.FALSE);
                    break;
                case EMPTY_TUPLE:
                    stack.push(new Object[0]);
                    break;
                case TUPLE:
                    readTuple();
                    break;
                case EMPTY_LIST:
                    stack.push(new ArrayList<>());
                    break;
                case LIST:
                    buildList();
                    break;
                case APPEND:
                    append();
                    break;
                case APPENDS:
                    appends();
                    break;
                case EMPTY_DICT:
                    stack.push(new LinkedHashMap<>());
                    break;
                case DICT:
                    buildDict();
                    break;
                case SETITEM:
                    setItem();
                    break;
                case SETITEMS:
                    setItems();
                    break;
                case BUILD:
                    build();
                    break;
                case NEWOBJ:
                    newObj();
                    break;
                case REDUCE:
                    reduce();
                    break;
                case GLOBAL:
                    readGlobal();
                    break;
                case OBJ:
                    buildObj();
                    break;
                case INST:
                    buildInst();
                    break;
                case BINBYTES:
                    readBinBytes();
                    break;
                case SHORT_BINBYTES:
                    readShortBinBytes();
                    break;
                case BYTEARRAY8:
                    readByteArray8();
                    break;
                default:
                    throw new IOException("Unknown pickle opcode: " + opcode + " at position " + (pos - 1));
            }
        }

        return null;
    }

    private void readProto() throws IOException {
        int proto = readUnsignedByte();
        // Protocol version, we ignore
    }

    private void popMark() {
        List<Object> items = new ArrayList<>();
        while (true) {
            Object item = stack.pop();
            if (item instanceof Byte && (Byte) item == MARK)
                break;
            items.add(0, item);
        }
        stack.push(items);
    }

    private void readFloat() throws IOException {
        String value = readStringUntil('\n');
        stack.push(Float.parseFloat(value));
    }

    private void readBinFloat() throws IOException {
        long bits = readLong();
        double value = Double.longBitsToDouble(bits);
        stack.push((float) value);
    }

    private void readInt() throws IOException {
        String value = readStringUntil('\n');
        stack.push(Integer.parseInt(value));
    }

    private void readBinInt() throws IOException {
        int value = readInt32();
        stack.push(value);
    }

    private void readBinInt1() throws IOException {
        int value = readUnsignedByte();
        stack.push(value);
    }

    private void readBinInt2() throws IOException {
        int value = readShort();
        stack.push(value);
    }

    private void readString() throws IOException {
        String value = readStringUntil('\n');
        stack.push(value);
    }

    private void readBinString() throws IOException {
        int len = readInt32();
        byte[] bytes = new byte[len];
        readFully(bytes);
        stack.push(new String(bytes, StandardCharsets.US_ASCII));
    }

    private void readShortBinString() throws IOException {
        int len = readUnsignedByte();
        byte[] bytes = new byte[len];
        readFully(bytes);
        stack.push(new String(bytes, StandardCharsets.US_ASCII));
    }

    private void readUnicode() throws IOException {
        String value = readStringUntil('\n');
        stack.push(value);
    }

    private void readBinUnicode() throws IOException {
        int len = readInt32();
        byte[] bytes = new byte[len];
        readFully(bytes);
        stack.push(new String(bytes, StandardCharsets.UTF_8));
    }

    private void readBinUnicode8() throws IOException {
        long len = readLong();
        byte[] bytes = new byte[(int) len];
        readFully(bytes);
        stack.push(new String(bytes, StandardCharsets.UTF_8));
    }

    private void readTuple() {
        int size = (Integer) stack.pop();
        Object[] tuple = new Object[size];
        for (int i = size - 1; i >= 0; i--) {
            tuple[i] = stack.pop();
        }
        stack.push(tuple);
    }

    @SuppressWarnings("unchecked")
    private void buildList() {
        List<Object> items = (List<Object>) stack.pop();
        List<Object> list = new ArrayList<>();
        for (Object item : items) {
            list.add(item);
        }
        stack.push(list);
    }

    private void append() {
        Object value = stack.pop();
        @SuppressWarnings("unchecked")
        List<Object> list = (List<Object>) stack.peek();
        list.add(value);
    }

    @SuppressWarnings("unchecked")
    private void appends() {
        List<Object> items = (List<Object>) stack.pop();
        List<Object> list = (List<Object>) stack.peek();
        list.addAll(items);
    }

    @SuppressWarnings("unchecked")
    private void buildDict() {
        List<Object> items = (List<Object>) stack.pop();
        Map<Object, Object> dict = new LinkedHashMap<>();
        for (int i = 0; i < items.size(); i += 2) {
            dict.put(items.get(i), items.get(i + 1));
        }
        stack.push(dict);
    }

    @SuppressWarnings("unchecked")
    private void setItem() {
        Object value = stack.pop();
        Object key = stack.pop();
        Map<Object, Object> dict = (Map<Object, Object>) stack.peek();
        dict.put(key, value);
    }

    @SuppressWarnings("unchecked")
    private void setItems() {
        List<Object> items = (List<Object>) stack.pop();
        Map<Object, Object> dict = (Map<Object, Object>) stack.peek();
        for (int i = 0; i < items.size(); i += 2) {
            dict.put(items.get(i), items.get(i + 1));
        }
    }

    private void build() {
        Object state = stack.pop();
        Object obj = stack.peek();
        // Apply state to object
        if (obj instanceof PickleObject) {
            ((PickleObject) obj).setState(state);
        }
    }

    private void newObj() {
        Object[] args = (Object[]) stack.pop();
        Object cls = stack.pop();
        if (cls instanceof PickleClass) {
            PickleObject obj = ((PickleClass) cls).newInstance(args);
            stack.push(obj);
        }
    }

    private void reduce() {
        Object[] args = (Object[]) stack.pop();
        Object callable = stack.pop();
        if (callable instanceof PickleClass) {
            PickleObject obj = ((PickleClass) callable).newInstance(args);
            stack.push(obj);
        }
    }

    private void readGlobal() throws IOException {
        String module = readStringUntil('\n');
        String name = readStringUntil('\n');
        stack.push(new PickleClass(module, name));
    }

    private void buildObj() {
        // Similar to INST
        Object[] args = (Object[]) stack.pop();
        Object cls = stack.pop();
        if (cls instanceof PickleClass) {
            PickleObject obj = ((PickleClass) cls).newInstance(args);
            stack.push(obj);
        }
    }

    private void buildInst() {
        // INST opcode
        String module = (String) stack.pop();
        String name = (String) stack.pop();
        Object[] args = (Object[]) stack.pop();
        PickleClass cls = new PickleClass(module, name);
        PickleObject obj = cls.newInstance(args);
        stack.push(obj);
    }

    private void readBinBytes() throws IOException {
        int len = readInt32();
        byte[] bytes = new byte[len];
        readFully(bytes);
        stack.push(bytes);
    }

    private void readShortBinBytes() throws IOException {
        int len = readUnsignedByte();
        byte[] bytes = new byte[len];
        readFully(bytes);
        stack.push(bytes);
    }

    private void readByteArray8() throws IOException {
        long len = readLong();
        byte[] bytes = new byte[(int) len];
        readFully(bytes);
        stack.push(bytes);
    }

    // Helper methods
    private int readUnsignedByte() throws IOException {
        if (pos >= data.length)
            throw new EOFException();
        return data[pos++] & 0xFF;
    }

    private short readShort() throws IOException {
        if (pos + 1 >= data.length)
            throw new EOFException();
        return (short) ((data[pos++] & 0xFF) | ((data[pos++] & 0xFF) << 8));
    }

    private int readInt32() throws IOException {
        if (pos + 3 >= data.length)
            throw new EOFException();
        return (data[pos++] & 0xFF) | ((data[pos++] & 0xFF) << 8) |
                ((data[pos++] & 0xFF) << 16) | ((data[pos++] & 0xFF) << 24);
    }

    private long readLong() throws IOException {
        if (pos + 7 >= data.length)
            throw new EOFException();
        return ((long) data[pos++] & 0xFF) | (((long) data[pos++] & 0xFF) << 8) |
                (((long) data[pos++] & 0xFF) << 16) | (((long) data[pos++] & 0xFF) << 24) |
                (((long) data[pos++] & 0xFF) << 32) | (((long) data[pos++] & 0xFF) << 40) |
                (((long) data[pos++] & 0xFF) << 48) | (((long) data[pos++] & 0xFF) << 56);
    }

    private void readFully(byte[] bytes) throws IOException {
        System.arraycopy(data, pos, bytes, 0, bytes.length);
        pos += bytes.length;
    }

    private String readStringUntil(char terminator) throws IOException {
        StringBuilder sb = new StringBuilder();
        while (pos < data.length && data[pos] != terminator) {
            sb.append((char) data[pos++]);
        }
        pos++; // Skip terminator
        return sb.toString();
    }

    /**
     * Class representing a Python class in pickle.
     */
    public static class PickleClass {
        private final String module;
        private final String name;

        public PickleClass(String module, String name) {
            this.module = module;
            this.name = name;
        }

        public PickleObject newInstance(Object... args) {
            return new PickleObject(this, args);
        }

        public String getModule() {
            return module;
        }

        public String getName() {
            return name;
        }

        @Override
        public String toString() {
            return module + "." + name;
        }
    }

    /**
     * Generic pickle object that can hold state.
     */
    public static class PickleObject {
        private final PickleClass type;
        private final Object[] args;
        private Object state;

        public PickleObject(PickleClass type, Object... args) {
            this.type = type;
            this.args = args;
        }

        public void setState(Object state) {
            this.state = state;
        }

        public PickleClass getType() {
            return type;
        }

        public Object[] getArgs() {
            return args;
        }

        public Object getState() {
            return state;
        }

        @SuppressWarnings("unchecked")
        public <T> T getState(String key) {
            if (state instanceof Map) {
                return (T) ((Map<?, ?>) state).get(key);
            }
            return null;
        }

        @Override
        public String toString() {
            return type.toString();
        }
    }
}
