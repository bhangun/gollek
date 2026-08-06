package tech.kayys.gollek.spi.inference;

/**
 * Represents the state of the Key-Value (KV) cache for an inference session.
 * Used for monitoring VRAM usage and managing offloading to CPU/NVMe.
 */
public interface KVCacheState {

    /**
     * @return Total VRAM allocated for this cache in bytes.
     */
    long getAllocatedVramBytes();

    /**
     * @return Total VRAM used by this cache in bytes.
     */
    long getUsedVramBytes();

    /**
     * @return The percentage of VRAM used (0.0 to 1.0).
     */
    double getVramUtilization();

    /**
     * @return true if the cache has been partially or fully offloaded.
     */
    boolean isOffloaded();

    /**
     * @return The size of the offloaded cache in bytes.
     */
    long getOffloadedBytes();
}
