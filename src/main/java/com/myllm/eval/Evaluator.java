package com.myllm.eval;

import com.myllm.crossentropy.CrossEntropyLoss;
import com.myllm.generator.Generator;
import com.myllm.generator.InteractiveSession;
import com.myllm.tokenize.Tokenizer;
import com.myllm.trainer.ChatExample;
import com.myllm.transformer.TransformerModel;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.regex.Pattern;

/**
 * Repeatable checks for a chat model, printed in the same format for every model so runs can be compared:
 * masked validation loss (forward pass only) and plot-word hit rates on fixed prompts.
 */
public final class Evaluator {

    // a fixed request and the plot words a reply should contain; each word may list alternatives ("puppy|dog")
    public record PlotPrompt(String request, String[] words) {}

    // phrased like the converter's templates, so they match what fine-tuning trained on
    public static final List<PlotPrompt> PLOT_PROMPTS = List.of(
            new PlotPrompt(
                    "Write a story with this plot: Tom finds a lost puppy in the park and takes it home.",
                    new String[] {"Tom", "puppy|dog", "park", "home"}),
            new PlotPrompt(
                    "Tell me a story. Here is the idea: Sara loses her red ball and her friend Ben helps her find it.",
                    new String[] {"Sara", "ball", "Ben", "find|found"}),
            new PlotPrompt(
                    "Write a story where this happens: A little bird is scared of the rain, but its mom keeps it warm.",
                    new String[] {"bird", "rain", "mom", "warm"}),
            new PlotPrompt(
                    "Can you write a short story based on this? Max the cat climbs a tall tree and cannot get down.",
                    new String[] {"Max", "cat", "tree", "down"}),
            new PlotPrompt(
                    "Write a story. Use the words cake, garden and happy.", new String[] {"cake", "garden", "happy"}));

    private Evaluator() {}

    public record LossResult(double loss, long answerTokens, int examples) {}

    // average loss per answer token over all examples (same mask as sft training), no weights change.
    // weighted by token count, so long answers count proportionally, like one big batch would
    public static LossResult validationLoss(TransformerModel model, List<ChatExample> examples) {
        CrossEntropyLoss loss = new CrossEntropyLoss();
        double total = 0;
        long tokens = 0;
        for (ChatExample example : examples) {
            int[] ids = example.ids();
            int[] input = Arrays.copyOfRange(ids, 0, ids.length - 1);
            int[] targets = Arrays.copyOfRange(ids, 1, ids.length);
            boolean[] mask = new boolean[targets.length];
            int counted = 0;
            for (int t = 0; t < mask.length; t++) {
                mask[t] = t + 1 >= example.answerStart();
                if (mask[t]) {
                    counted++;
                }
            }
            total += loss.forwardPass(model.forwardPass(input), targets, mask) * counted;
            tokens += counted;
        }
        return new LossResult(total / tokens, tokens, examples.size());
    }

    // samples replies for every fixed prompt and reports how many contain each plot word; plot score = average
    // fraction of plot words per reply over all prompts. Returns the plot score
    public static double plotReport(
            TransformerModel model,
            Tokenizer tokenizer,
            int samples,
            int length,
            int topK,
            double temperature,
            long seed,
            PrintStream out) {
        Generator generator = new Generator(model, tokenizer, new Random(seed));
        double scoreSum = 0;
        int replies = 0;
        for (PlotPrompt prompt : PLOT_PROMPTS) {
            int[] hits = new int[prompt.words().length];
            int stopped = 0;
            String firstReply = null;
            for (int s = 0; s < samples; s++) {
                Generator.Completion reply = generator.complete(
                        InteractiveSession.chatPrompt(prompt.request()),
                        length,
                        topK,
                        temperature,
                        InteractiveSession.NEXT_TURN);
                if (firstReply == null) {
                    firstReply = reply.text().strip();
                }
                if (reply.stopped()) {
                    stopped++;
                }
                int found = 0;
                for (int w = 0; w < hits.length; w++) {
                    if (containsWord(reply.text(), prompt.words()[w])) {
                        hits[w]++;
                        found++;
                    }
                }
                scoreSum += (double) found / hits.length;
                replies++;
            }
            out.printf("%n> %s%n", prompt.request());
            StringBuilder counts = new StringBuilder();
            for (int w = 0; w < hits.length; w++) {
                counts.append(String.format("%s %d/%d  ", prompt.words()[w], hits[w], samples));
            }
            out.printf("  %sstopped %d/%d%n", counts, stopped, samples);
            out.printf("  sample: %s%n", oneLine(firstReply, 300));
        }
        return scoreSum / replies;
    }

    // whole-word, case-insensitive match of any alternative, so "park" doesn't count inside "sparkle"
    static boolean containsWord(String text, String alternatives) {
        for (String word : alternatives.split("\\|")) {
            Pattern pattern = Pattern.compile("\\b" + Pattern.quote(word.toLowerCase(Locale.ROOT)) + "\\b");
            if (pattern.matcher(text.toLowerCase(Locale.ROOT)).find()) {
                return true;
            }
        }
        return false;
    }

    private static String oneLine(String text, int maxChars) {
        String flat = text.replace('\n', ' ');
        return flat.length() <= maxChars ? flat : flat.substring(0, maxChars) + "...";
    }
}
