package com.myllm.server;

import java.util.HashMap;
import java.util.Map;

/**
 * Just enough JSON for the chat API: reads one flat object of strings, numbers and booleans, and escapes
 * strings for responses. No nesting and no arrays; anything else is rejected.
 */
final class Json {

    private Json() {}

    static Map<String, Object> parseObject(String text) {
        Parser p = new Parser(text);
        Map<String, Object> result = new HashMap<>();
        p.skip();
        p.expect('{');
        p.skip();
        if (p.peek() == '}') {
            p.pos++;
            return result;
        }
        while (true) {
            p.skip();
            String key = p.string();
            p.skip();
            p.expect(':');
            p.skip();
            result.put(key, p.value());
            p.skip();
            char c = p.next();
            if (c == '}') {
                return result;
            }
            if (c != ',') {
                throw new IllegalArgumentException("expected , or } at " + (p.pos - 1));
            }
        }
    }

    static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.append('"').toString();
    }

    private static final class Parser {
        private final String s;
        private int pos;

        Parser(String s) {
            this.s = s;
        }

        char peek() {
            if (pos >= s.length()) {
                throw new IllegalArgumentException("unexpected end of JSON");
            }
            return s.charAt(pos);
        }

        char next() {
            char c = peek();
            pos++;
            return c;
        }

        void skip() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        void expect(char c) {
            if (next() != c) {
                throw new IllegalArgumentException("expected " + c + " at " + (pos - 1));
            }
        }

        Object value() {
            char c = peek();
            if (c == '"') {
                return string();
            }
            if (s.startsWith("true", pos)) {
                pos += 4;
                return true;
            }
            if (s.startsWith("false", pos)) {
                pos += 5;
                return false;
            }
            int start = pos;
            while (pos < s.length() && "+-0123456789.eE".indexOf(s.charAt(pos)) >= 0) {
                pos++;
            }
            if (start == pos) {
                throw new IllegalArgumentException("unsupported JSON value at " + start);
            }
            return Double.parseDouble(s.substring(start, pos));
        }

        String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = next();
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                char e = next();
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case 'r' -> sb.append('\r');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (pos + 4 > s.length()) {
                            throw new IllegalArgumentException("bad \\u escape");
                        }
                        sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> sb.append(e); // \" \\ \/
                }
            }
        }
    }
}
