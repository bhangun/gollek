package tech.kayys.gollek.ml.automl;

import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.optim.Optimizer;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Random-search hyperparameter optimization — equivalent to
 * {@code sklearn.model_selection.RandomizedSearchCV} for deep learning.
 *
 * <p>Samples hyperparameter configurations from defined search spaces,
 * trains a model for each configuration, and returns the best one.
 * Uses JDK 25 virtual threads to run trials in parallel.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var hpo = HyperparameterSearch.builder()
 *     .addFloat("lr",      1e-4f, 1e-2f)
 *     .addInt("batchSize", 16, 128)
 *     .addChoice("optimizer", "adam", "sgd", "rmsprop")
 *     .trials(20)
 *     .parallelTrials(4)
 *     .objective((config, trial) -> trainAndEval(config))
 *     .build();
 *
 * HyperparameterSearch.Result best = hpo.run();
 * System.out.println("Best config: " + best.config());
 * System.out.println("Best score:  " + best.score());
 * }</pre>
 */
public final class HyperparameterSearch {

    /**
     * A single hyperparameter configuration: name → value.
     * Values are {@code Float}, {@code Integer}, or {@code String}.
     */
    public record Config(Map<String, Object> params) {

        /**
         * Returns a float hyperparameter value.
         *
         * @param name parameter name
         * @return float value
         */
        public float getFloat(String name) { return ((Number) params.get(name)).floatValue(); }

        /**
         * Returns an integer hyperparameter value.
         *
         * @param name parameter name
         * @return int value
         */
        public int getInt(String name) { return ((Number) params.get(name)).intValue(); }

        /**
         * Returns a string (categorical) hyperparameter value.
         *
         * @param name parameter name
         * @return string value
         */
        public String getString(String name) { return (String) params.get(name); }
    }

    /**
     * Result of a completed HPO run.
     *
     * @param config best hyperparameter configuration found
     * @param score  best objective score (higher is better)
     * @param allResults all trial results sorted by score descending
     */
    public record Result(Config config, float score, List<TrialResult> allResults) {}

    /**
     * Result of a single trial.
     *
     * @param config configuration used
     * @param score  objective score achieved
     * @param trialId trial index
     */
    public record TrialResult(Config config, float score, int trialId) {}

    private final List<SearchSpace> spaces;
    private final int trials;
    private final int parallelTrials;
    private final BiFunction<Config, Integer, Float> objective;
    private final Random rng = new Random();

    private HyperparameterSearch(Builder b) {
        this.spaces        = b.spaces;
        this.trials        = b.trials;
        this.parallelTrials = b.parallelTrials;
        this.objective     = b.objective;
    }

    /**
     * Runs the hyperparameter search.
     *
     * <p>Samples {@code trials} configurations and evaluates them using
     * virtual threads (up to {@code parallelTrials} concurrently).
     *
     * @return {@link Result} with the best configuration and all trial results
     */
    public Result run() {
        List<Config> configs = new ArrayList<>(trials);
        for (int i = 0; i < trials; i++) configs.add(sample());

        List<TrialResult> results = new CopyOnWriteArrayList<>();
        try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
            Semaphore sem = new Semaphore(parallelTrials);
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < configs.size(); i++) {
                final int trialId = i;
                final Config cfg  = configs.get(i);
                futures.add(exec.submit(() -> {
                    try {
                        sem.acquire();
                        float score = objective.apply(cfg, trialId);
                        results.add(new TrialResult(cfg, score, trialId));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        sem.release();
                    }
                }));
            }
            for (Future<?> f : futures) {
                try { f.get(); } catch (Exception e) { /* log and continue */ }
            }
        }

        results.sort(Comparator.comparingDouble(TrialResult::score).reversed());
        TrialResult best = results.isEmpty()
            ? new TrialResult(configs.get(0), Float.NEGATIVE_INFINITY, 0)
            : results.get(0);
        return new Result(best.config(), best.score(), Collections.unmodifiableList(results));
    }

    /** Samples one random configuration from all search spaces. */
    private Config sample() {
        Map<String, Object> params = new LinkedHashMap<>();
        for (SearchSpace s : spaces) params.put(s.name(), s.sample(rng));
        return new Config(params);
    }

    /** @return a new builder */
    public static Builder builder() { return new Builder(); }

    // ── Search space definitions ──────────────────────────────────────────

    private sealed interface SearchSpace permits FloatSpace, IntSpace, ChoiceSpace {
        String name();
        Object sample(Random rng);
    }

    private record FloatSpace(String name, float lo, float hi) implements SearchSpace {
        public Object sample(Random rng) { return lo + rng.nextFloat() * (hi - lo); }
    }

    private record IntSpace(String name, int lo, int hi) implements SearchSpace {
        public Object sample(Random rng) { return lo + rng.nextInt(hi - lo + 1); }
    }

    private record ChoiceSpace(String name, String[] choices) implements SearchSpace {
        public Object sample(Random rng) { return choices[rng.nextInt(choices.length)]; }
    }

    /**
     * Builder for {@link HyperparameterSearch}.
     */
    public static final class Builder {
        private final List<SearchSpace> spaces = new ArrayList<>();
        private int trials = 10;
        private int parallelTrials = 2;
        private BiFunction<Config, Integer, Float> objective;

        /**
         * Adds a continuous float hyperparameter sampled uniformly from {@code [lo, hi]}.
         *
         * @param name parameter name
         * @param lo   lower bound (inclusive)
         * @param hi   upper bound (exclusive)
         */
        public Builder addFloat(String name, float lo, float hi) {
            spaces.add(new FloatSpace(name, lo, hi)); return this;
        }

        /**
         * Adds an integer hyperparameter sampled uniformly from {@code [lo, hi]}.
         *
         * @param name parameter name
         * @param lo   lower bound (inclusive)
         * @param hi   upper bound (inclusive)
         */
        public Builder addInt(String name, int lo, int hi) {
            spaces.add(new IntSpace(name, lo, hi)); return this;
        }

        /**
         * Adds a categorical hyperparameter chosen uniformly from {@code choices}.
         *
         * @param name    parameter name
         * @param choices possible string values
         */
        public Builder addChoice(String name, String... choices) {
            spaces.add(new ChoiceSpace(name, choices)); return this;
        }

        /** @param n total number of trials to run */
        public Builder trials(int n)           { this.trials = n; return this; }

        /** @param n maximum concurrent trials (uses virtual threads) */
        public Builder parallelTrials(int n)   { this.parallelTrials = n; return this; }

        /**
         * Sets the objective function: {@code (config, trialId) → score}.
         * Higher score = better. Typically returns validation accuracy or negative loss.
         *
         * @param fn objective function
         */
        public Builder objective(BiFunction<Config, Integer, Float> fn) {
            this.objective = fn; return this;
        }

        /**
         * Builds the {@link HyperparameterSearch}.
         *
         * @return configured search
         * @throws IllegalStateException if objective is not set
         */
        public HyperparameterSearch build() {
            if (objective == null) throw new IllegalStateException("objective must be set");
            return new HyperparameterSearch(this);
        }
    }
}
