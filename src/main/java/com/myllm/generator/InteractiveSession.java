package com.myllm.generator;

import com.myllm.tokenize.Tokenizer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Set;

/**
 * Reads prompts line by line and prints the model's continuation. Ends on {@code /quit} or end of input (Ctrl+D).
 *
 * <p>With the chat template, each line is sent as {@code "User: <line>\nAssistant:"} (the format fine-tuning
 * used) and only the reply is printed.
 */
public class InteractiveSession {

    private static final String QUIT = "/quit";
    public static final String USER_PREFIX = "User: ";
    public static final String ASSISTANT_PREFIX = "\nAssistant:";
    // a chat model may start writing the next user turn itself; the reply ends there
    public static final String NEXT_TURN = "\nUser:";

    private final Generator generator;
    private final Tokenizer tokenizer;
    private final int length;
    private final int topK;
    private final double temperature;
    private final boolean chatTemplate;

    // greedy decoding
    public InteractiveSession(Generator generator, Tokenizer tokenizer, int length) {
        this(generator, tokenizer, length, 0, 0);
    }

    // temperature <= 0 means greedy, topK <= 0 means no top-k filter
    public InteractiveSession(Generator generator, Tokenizer tokenizer, int length, int topK, double temperature) {
        this(generator, tokenizer, length, topK, temperature, false);
    }

    public InteractiveSession(
            Generator generator, Tokenizer tokenizer, int length, int topK, double temperature, boolean chatTemplate) {
        this.generator = generator;
        this.tokenizer = tokenizer;
        this.length = length;
        this.topK = topK;
        this.temperature = temperature;
        this.chatTemplate = chatTemplate;
    }

    // the exact prompt fine-tuning trained on, so the model sees the same "Assistant:" boundary
    public static String chatPrompt(String request) {
        return USER_PREFIX + request + ASSISTANT_PREFIX;
    }

    public void run(BufferedReader in, PrintStream out) throws IOException {
        out.printf(
                "model ready (vocab %d tokens%s). type a prompt, %s or Ctrl+D to exit.%n",
                tokenizer.vocabSize(), chatTemplate ? ", chat template on" : "", QUIT);
        while (true) {
            out.print("> ");
            out.flush();
            String line = in.readLine();
            if (line == null || line.strip().equals(QUIT)) {
                out.println();
                return;
            }
            if (line.isEmpty()) {
                continue;
            }
            String prompt = chatTemplate ? chatPrompt(line) : line;
            Set<Character> unknown = tokenizer.unknownChars(prompt);
            if (!unknown.isEmpty()) {
                // model can only read characters it saw during training
                out.println("not in model vocab: " + unknown);
                continue;
            }
            if (chatTemplate) {
                out.println(generator
                        .complete(prompt, length, topK, temperature, NEXT_TURN)
                        .text()
                        .strip());
            } else {
                out.println(generator.generate(prompt, length, topK, temperature));
            }
        }
    }
}
