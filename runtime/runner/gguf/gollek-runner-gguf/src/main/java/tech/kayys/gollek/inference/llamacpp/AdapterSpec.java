package tech.kayys.gollek.inference.llamacpp;

public record AdapterSpec(String type, String adapterId, String adapterPath, double scale) {
    public boolean isType(String t) {
        return type != null && type.equalsIgnoreCase(t);
    }
    
    public String cacheKey() {
        return adapterId + ":" + scale;
    }
}
