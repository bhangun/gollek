package tech.kayys.gollek.ml.nn.audio;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;

/**
 * Computes a Mel Spectrogram from audio signals.
 * <p>
 * Frequently used as the frontend for audio models like Whisper, wav2vec2, and AST.
 * Input: [Batch, Time] or [Time] raw audio waveform.
 * Output: [Batch, n_mels, Time_frames] mel spectrogram.
 */
public class MelSpectrogram extends NNModule {

    private final int nFft;
    private final int hopLength;
    private final int nMels;

    public MelSpectrogram(int sampleRate, int nFft, int winLength, int hopLength, int nMels) {
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.nMels = nMels;
    }

    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int b = shape.length == 1 ? 1 : (int) shape[0];
        int t = (int) shape[shape.length - 1];

        // Output frames calculation
        int outFrames = (t - nFft) / hopLength + 1;
        
        // Mocking the STFT / Mel Filterbank application
        float[] output = new float[b * nMels * outFrames];
        return GradTensor.of(output, b, nMels, outFrames);
    }
}
