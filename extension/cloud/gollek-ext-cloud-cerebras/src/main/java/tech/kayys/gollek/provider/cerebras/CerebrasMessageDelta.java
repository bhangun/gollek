package tech.kayys.gollek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Delta content in a streaming choice for Cerebras.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class CerebrasMessageDelta {

        @JsonProperty("role")
        private String role;

        @JsonProperty("content")
        private String content;

        public String getRole() {
                return role;
        }

        public void setRole(String role) {
                this.role = role;
        }

        public String getContent() {
                return content;
        }

        public void setContent(String content) {
                this.content = content;
        }
}