package com.myllm.server;

import com.myllm.generator.Generator;
import com.myllm.generator.InteractiveSession;
import com.myllm.persist.ModelIO;
import com.myllm.tokenize.Tokenizer;
import com.myllm.transformer.TransformerModel;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Local web chat: serves {@code web/chat.html} and a small JSON API around {@code Generator}. Listens on
 * 127.0.0.1 only.
 * <ul>
 *   <li>{@code GET /}: the page
 *   <li>{@code GET /api/info}: the loaded model and every {@code .bin} in the models directory
 *   <li>{@code POST /api/load}, body {@code {"model": "<file name>"}}: switch model
 *   <li>{@code POST /api/tokens}, body {@code {"text": "...", "mode": "chat"|"complete"}}: prompt token count
 *   <li>{@code POST /api/generate}, body fields {@code message, mode, maxTokens, temperature, topK}: one reply
 * </ul>
 * The model keeps forward caches, so requests are served one at a time ({@code synchronized}).
 */
public class ChatServer {

    private static final int MAX_TOKENS_LIMIT = 1024;
    private static final int MAX_BODY_BYTES = 64 * 1024;

    private final Path modelsDir;
    private Path modelPath;
    private TransformerModel model;
    private Tokenizer tokenizer;
    private Generator generator;

    public ChatServer(Path modelPath) throws IOException {
        Path absolute = modelPath.toAbsolutePath().normalize();
        this.modelsDir = absolute.getParent();
        load(absolute);
    }

    private synchronized void load(Path path) throws IOException {
        ModelIO.LoadedModel loaded = ModelIO.load(path);
        model = loaded.model();
        tokenizer = loaded.tokenizer();
        generator = new Generator(model, tokenizer, new Random());
        modelPath = path;
    }

    public void start(int port) throws IOException {
        // loopback only: anyone who can reach the port can make the model generate
        HttpServer server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        server.createContext("/", this::page);
        server.createContext("/api/info", exchange -> handle(exchange, "GET", body -> info()));
        server.createContext("/api/load", exchange -> handle(exchange, "POST", this::loadRequest));
        server.createContext("/api/tokens", exchange -> handle(exchange, "POST", this::tokens));
        server.createContext("/api/generate", exchange -> handle(exchange, "POST", this::generate));
        server.start();
        System.out.printf(
                "chat UI on http://localhost:%d  (model %s, Ctrl+C to stop)%n", port, modelPath.getFileName());
    }

    private interface Handler {
        String apply(Map<String, Object> body) throws IOException;
    }

    private void handle(HttpExchange exchange, String method, Handler handler) throws IOException {
        try (exchange) {
            if (!exchange.getRequestMethod().equals(method)) {
                send(exchange, 405, "{\"error\":\"use " + method + "\"}");
                return;
            }
            try {
                Map<String, Object> body = method.equals("POST") ? Json.parseObject(readBody(exchange)) : Map.of();
                send(exchange, 200, handler.apply(body));
            } catch (IllegalArgumentException e) {
                send(exchange, 400, "{\"error\":" + Json.quote(String.valueOf(e.getMessage())) + "}");
            } catch (IOException | RuntimeException e) {
                send(exchange, 500, "{\"error\":" + Json.quote(e.toString()) + "}");
            }
        }
    }

    private static String readBody(HttpExchange exchange) throws IOException {
        try (InputStream in = exchange.getRequestBody()) {
            byte[] bytes = in.readNBytes(MAX_BODY_BYTES + 1);
            if (bytes.length > MAX_BODY_BYTES) {
                throw new IllegalArgumentException("request too large");
            }
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    private static void send(HttpExchange exchange, int status, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private void page(HttpExchange exchange) throws IOException {
        try (exchange) {
            if (!exchange.getRequestURI().getPath().equals("/")) {
                exchange.sendResponseHeaders(404, -1);
                return;
            }
            try (InputStream in = ChatServer.class.getResourceAsStream("/web/chat.html")) {
                if (in == null) {
                    byte[] msg = "web/chat.html not on the classpath; run with build/resources/main on -cp"
                            .getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(500, msg.length);
                    exchange.getResponseBody().write(msg);
                    return;
                }
                byte[] html = in.readAllBytes();
                exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
                exchange.sendResponseHeaders(200, html.length);
                exchange.getResponseBody().write(html);
            }
        }
    }

    private synchronized String info() throws IOException {
        TransformerModel.ModelConfig c = model.getConfig();
        long parameters = model.parameters().stream()
                .mapToLong(m -> (long) m.rows() * m.cols())
                .sum();
        List<String> models;
        try (Stream<Path> files = Files.list(modelsDir)) {
            models = files.map(p -> p.getFileName().toString())
                    .filter(n -> n.endsWith(".bin"))
                    .sorted()
                    .toList();
        }
        return "{\"model\":" + Json.quote(modelPath.getFileName().toString())
                + ",\"vocab\":" + c.vocabSize() + ",\"tMax\":" + c.T_max() + ",\"dModel\":" + c.d_Model()
                + ",\"heads\":" + c.heads() + ",\"blocks\":" + c.noOfDecoderBlocks() + ",\"parameters\":" + parameters
                + ",\"models\":[" + models.stream().map(Json::quote).collect(Collectors.joining(",")) + "]}";
    }

    private String loadRequest(Map<String, Object> body) throws IOException {
        String name = string(body, "model");
        // a bare file name from the list, never a path, so the page can't load files outside the models directory
        if (name.contains("/") || name.contains("\\") || !name.endsWith(".bin")) {
            throw new IllegalArgumentException("model must be a .bin file name from the list");
        }
        Path path = modelsDir.resolve(name);
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("no such model: " + name);
        }
        load(path);
        return info();
    }

    private synchronized String tokens(Map<String, Object> body) {
        String prompt = prompt(body, "text");
        Set<Character> unknown = tokenizer.unknownChars(prompt);
        int count = unknown.isEmpty() ? tokenizer.encode(prompt).length : -1;
        return "{\"tokens\":" + count + ",\"unknown\":" + Json.quote(unknown.toString()) + "}";
    }

    private synchronized String generate(Map<String, Object> body) {
        boolean chat = isChat(body);
        String prompt = prompt(body, "message");
        if (string(body, "message").isBlank()) {
            throw new IllegalArgumentException("message is empty");
        }
        Set<Character> unknown = tokenizer.unknownChars(prompt);
        if (!unknown.isEmpty()) {
            // the model can only read characters it saw during training
            throw new IllegalArgumentException("not in model vocab: " + unknown);
        }
        int maxTokens = (int) number(body, "maxTokens", 200);
        if (maxTokens < 1 || maxTokens > MAX_TOKENS_LIMIT) {
            throw new IllegalArgumentException("maxTokens must be 1.." + MAX_TOKENS_LIMIT);
        }
        double temperature = number(body, "temperature", 0.8);
        int topK = (int) number(body, "topK", 10);
        int promptTokens = tokenizer.encode(prompt).length;
        long start = System.nanoTime();
        Generator.Completion completion =
                generator.complete(prompt, maxTokens, topK, temperature, chat ? InteractiveSession.NEXT_TURN : null);
        long ms = (System.nanoTime() - start) / 1_000_000;
        String reply = chat ? completion.text().strip() : completion.text();
        int replyTokens = completion.text().isEmpty() ? 0 : tokenizer.encode(completion.text()).length;
        return "{\"reply\":" + Json.quote(reply) + ",\"promptTokens\":" + promptTokens + ",\"replyTokens\":"
                + replyTokens + ",\"stopped\":" + completion.stopped() + ",\"ms\":" + ms + ",\"tMax\":"
                + model.getConfig().T_max() + "}";
    }

    private static boolean isChat(Map<String, Object> body) {
        return !"complete".equals(body.getOrDefault("mode", "chat"));
    }

    // chat mode wraps the text in the template fine-tuning used; complete mode continues the raw text
    private static String prompt(Map<String, Object> body, String field) {
        String text = string(body, field);
        return isChat(body) ? InteractiveSession.chatPrompt(text) : text;
    }

    private static String string(Map<String, Object> body, String key) {
        Object value = body.get(key);
        if (!(value instanceof String s)) {
            throw new IllegalArgumentException(key + " must be a string");
        }
        return s;
    }

    private static double number(Map<String, Object> body, String key, double fallback) {
        Object value = body.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof Double d) || d.isNaN() || d.isInfinite()) {
            throw new IllegalArgumentException(key + " must be a number");
        }
        return d;
    }
}
