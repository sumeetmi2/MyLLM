package com.myllm.generator;

import com.myllm.dao.Matrix;
import com.myllm.tokenize.Tokenizer;
import com.myllm.transformer.TransformerModel;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * Text generation: repeatedly predicts the next token and feeds it back in. Forward pass only, no training.
 *
 * <p>Greedy when {@code temperature <= 0}, otherwise temperature plus top-k sampling.
 */
public class Generator {

    private final TransformerModel model;
    private final Tokenizer tokenizer;
    private final Random random;

    public Generator(TransformerModel model, Tokenizer tokenizer) {
        this(model, tokenizer, new Random());
    }

    // pass a seeded Random for reproducible sampling (tests)
    public Generator(TransformerModel model, Tokenizer tokenizer, Random random) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.random = random;
    }

    // greedy: returns prompt + up to n generated tokens
    public String generate(String prompt, int n) {
        return generate(prompt, n, 0, 0);
    }

    // returns prompt + up to n generated tokens; temperature <= 0 means greedy, k <= 0 means no top-k filter.
    // Stops early when the model emits the tokenizer's end-of-text token (not included in the output).
    public String generate(String prompt, int n, int k, double temperature) {
        return prompt + complete(prompt, n, k, temperature, null).text();
    }

    // text = generated part only (prompt not included); stopped = ended before n tokens, by end-of-text or stop
    public record Completion(String text, boolean stopped) {}

    // like generate, but returns only the generated text. stop (may be null) also ends generation as soon as the
    // generated text contains it, and is cut off with everything after it, e.g. "\nUser:" when a chat model
    // starts writing the next user turn itself
    public Completion complete(String prompt, int n, int k, double temperature, String stop) {
        if (prompt == null || prompt.isEmpty()) {
            throw new IllegalArgumentException("prompt must contain at least one character");
        }
        if (n < 0) {
            throw new IllegalArgumentException("n must be >= 0, got " + n);
        }
        int[] promptIds = tokenizer.encode(prompt);
        int[] context = Arrays.copyOf(promptIds, promptIds.length + n);
        int length = promptIds.length;
        int tMax = model.getEmbedding().getTMax();
        int endOfText = tokenizer.endOfTextId();

        for (int i = 0; i < n; i++) {
            // model can't take more than T_max positions, so only the most recent T_max tokens are fed in
            int start = Math.max(0, length - tMax);
            Matrix logits = model.forwardPass(Arrays.copyOfRange(context, start, length));
            // last row predicts the token after the whole window; earlier rows predict tokens we already have
            double[] row = new double[logits.cols()];
            for (int j = 0, last = logits.offset(logits.rows() - 1); j < row.length; j++) {
                row[j] = logits.data()[last + j];
            }
            int next = sample(row, k, temperature, random);
            if (next == endOfText) {
                // the model says the text is finished
                return new Completion(generatedText(context, promptIds.length, length), true);
            }
            context[length++] = next;
            if (stop != null) {
                String text = generatedText(context, promptIds.length, length);
                int at = text.indexOf(stop);
                if (at >= 0) {
                    return new Completion(text.substring(0, at), true);
                }
            }
        }
        return new Completion(generatedText(context, promptIds.length, length), false);
    }

    private String generatedText(int[] context, int from, int to) {
        return tokenizer.decode(Arrays.copyOfRange(context, from, to));
    }

    // picks the next token id from one row of logits
    static int sample(double[] row, int k, double temperature, Random random) {
        if (temperature <= 0) {
            return argmax(row);
        }
        if (k <= 0 || k > row.length) {
            k = row.length;
        }
        // top-k: queue holds vocab indices, ordered by their logit, largest first
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> Double.compare(row[b], row[a]));
        for (int i = 0; i < row.length; i++) {
            pq.add(i);
        }
        int[] kept = new int[k];
        double[] cumulative = new double[k];
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            kept[i] = pq.poll();
            // kept[0] has the largest logit, so every exponent is <= 0 and exp can't overflow
            cumulative[i] = Math.exp((row[kept[i]] - row[kept[0]]) / temperature);
            sum += cumulative[i];
        }
        // normalise to probabilities, then running total
        for (int i = 0; i < k; i++) {
            cumulative[i] = cumulative[i] / sum;
            if (i > 0) {
                cumulative[i] += cumulative[i - 1];
            }
        }
        // weighted dice roll: insertion point = first index whose cumulative value exceeds r
        int found = Arrays.binarySearch(cumulative, random.nextDouble());
        if (found < 0) {
            found = -(found + 1);
        }
        // rounding can leave the last cumulative value slightly below 1
        found = Math.min(found, k - 1);
        return kept[found];
    }

    // softmax keeps the ordering of scores, so argmax over logits equals argmax over probabilities
    static int argmax(double[] scores) {
        int best = 0;
        for (int j = 1; j < scores.length; j++) {
            if (scores[j] > scores[best]) {
                best = j;
            }
        }
        return best;
    }
}
