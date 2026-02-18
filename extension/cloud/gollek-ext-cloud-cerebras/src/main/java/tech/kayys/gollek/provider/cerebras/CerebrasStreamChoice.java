package tech.kayys.gollek.provider.cerebras;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Cerebras streaming choice DTO
 */
public class CerebrasStreamChoice {

        private int index;
        private CerebrasMessageDelta delta;

        @JsonProperty("finish_reason")
        private String finishReason;

        public int getIndex() {
                return index;
        }

        public void setIndex(int index) {
                this.index = index;
        }

        public CerebrasMessageDelta getDelta() {
                return delta;
        }

        public void setDelta(CerebrasMessageDelta delta) {
                this.delta = delta;
        }

        public String getFinishReason() {
                return finishReason;
        }

        public void setFinishReason(String finishReason) {
                this.finishReason = finishReason;
        }
}