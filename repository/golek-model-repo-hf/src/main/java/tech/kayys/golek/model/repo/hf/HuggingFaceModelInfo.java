package tech.kayys.golek.model.repository.hf;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * HuggingFace model metadata
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class HuggingFaceModelInfo {

    @JsonProperty("id")
    private String modelId;

    @JsonProperty("modelId")
    private String alternateModelId;

    @JsonProperty("sha")
    private String commitSha;

    @JsonProperty("lastModified")
    private Instant lastModified;

    @JsonProperty("private")
    private boolean isPrivate;

    @JsonProperty("downloads")
    private long downloads;

    @JsonProperty("likes")
    private long likes;

    @JsonProperty("tags")
    private List<String> tags;

    @JsonProperty("pipeline_tag")
    private String pipelineTag;

    @JsonProperty("library_name")
    private String libraryName;

    @JsonProperty("siblings")
    private List<ModelFile> files;

    @JsonProperty("config")
    private Map<String, Object> config;

    // Getters and setters
    public String getModelId() {
        return modelId != null ? modelId : alternateModelId;
    }

    public void setModelId(String modelId) {
        this.modelId = modelId;
    }

    public String getCommitSha() {
        return commitSha;
    }

    public void setCommitSha(String commitSha) {
        this.commitSha = commitSha;
    }

    public Instant getLastModified() {
        return lastModified;
    }

    public void setLastModified(Instant lastModified) {
        this.lastModified = lastModified;
    }

    public boolean isPrivate() {
        return isPrivate;
    }

    public void setPrivate(boolean isPrivate) {
        this.isPrivate = isPrivate;
    }

    public long getDownloads() {
        return downloads;
    }

    public void setDownloads(long downloads) {
        this.downloads = downloads;
    }

    public long getLikes() {
        return likes;
    }

    public void setLikes(long likes) {
        this.likes = likes;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public String getPipelineTag() {
        return pipelineTag;
    }

    public void setPipelineTag(String pipelineTag) {
        this.pipelineTag = pipelineTag;
    }

    public String getLibraryName() {
        return libraryName;
    }

    public void setLibraryName(String libraryName) {
        this.libraryName = libraryName;
    }

    public List<ModelFile> getFiles() {
        return files;
    }

    public void setFiles(List<ModelFile> files) {
        this.files = files;
    }

    public Map<String, Object> getConfig() {
        return config;
    }

    public void setConfig(Map<String, Object> config) {
        this.config = config;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ModelFile {
        @JsonProperty("rfilename")
        private String filename;

        @JsonProperty("size")
        private Long size;

        @JsonProperty("lfs")
        private LFSInfo lfs;

        public String getFilename() {
            return filename;
        }

        public void setFilename(String filename) {
            this.filename = filename;
        }

        public Long getSize() {
            return size;
        }

        public void setSize(Long size) {
            this.size = size;
        }

        public LFSInfo getLfs() {
            return lfs;
        }

        public void setLfs(LFSInfo lfs) {
            this.lfs = lfs;
        }

        public boolean isLFS() {
            return lfs != null;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LFSInfo {
        @JsonProperty("size")
        private Long size;

        @JsonProperty("sha256")
        private String sha256;

        @JsonProperty("pointer_size")
        private Long pointerSize;

        public Long getSize() {
            return size;
        }

        public void setSize(Long size) {
            this.size = size;
        }

        public String getSha256() {
            return sha256;
        }

        public void setSha256(String sha256) {
            this.sha256 = sha256;
        }

        public Long getPointerSize() {
            return pointerSize;
        }

        public void setPointerSize(Long pointerSize) {
            this.pointerSize = pointerSize;
        }
    }
}