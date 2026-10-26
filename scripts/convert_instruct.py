"""Converts TinyStories-Instruct entries into single-turn chat examples for fine-tuning.

Input entries (fields in any order, separated by <|endoftext|>):
    Features: Dialogue, BadEnding
    Words: oak, gloomy, quit
    Summary: Ben wants to climb the big oak tree, but Lily is scared.
    Random sentence: The sky was gloomy and gray.
    Story:
    Once upon a time...

Output (one example per entry):
    User: Write a story with this plot: Ben wants to climb ... It should have dialogue.
    Assistant: Once upon a time...
    <|endoftext|>

Usage: python3 scripts/convert_instruct.py <input.txt> <output.txt> [--max-chars 700] [--seed 42]
"""

import argparse
import random
import re

END_OF_TEXT = "<|endoftext|>"
FIELDS = ("Features", "Words", "Summary", "Random sentence", "Story")
FIELD_LINE = re.compile(r"^(Features|Words|Summary|Random sentence|Story):\s*(.*)$")

# each feature becomes one sentence of the request
FEATURE_PHRASES = {
    "Dialogue": "It should have dialogue.",
    "BadEnding": "It should have a sad ending.",
    "MoralValue": "It should teach a lesson.",
    "Twist": "Add a surprise twist.",
    "Foreshadowing": "Use some foreshadowing.",
    "Conflict": "Include a conflict.",
}

# the summary is a full sentence with names in it, so templates take it as-is instead of "a story about <summary>"
PLOT_TEMPLATES = (
    "Write a story with this plot: {}",
    "Tell me a story. Here is the idea: {}",
    "Can you write a short story based on this? {}",
    "Write a story where this happens: {}",
)
NO_PLOT_TEMPLATES = (
    "Write a story.",
    "Tell me a story.",
    "Can you write a short story?",
)

# chance each part of the request is kept; dropping some teaches the model to handle short requests too
KEEP_SUMMARY = 0.8
KEEP_WORDS = 0.7
KEEP_FEATURES = 0.7

# typographic characters the Java CorpusCleaner would map anyway; done here so --max-chars counts the final text
ASCII_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                           "–": "-", "—": "-", "…": "...", " ": " "})


def parse_entry(entry):
    """Returns {field: text}. Story keeps its line breaks; it ends where the next field line starts."""
    fields = {}
    current = None
    for line in entry.strip().split("\n"):
        match = FIELD_LINE.match(line)
        if match:
            current = match.group(1)
            fields[current] = match.group(2).strip()
        elif current == "Story":
            fields["Story"] += "\n" + line
        elif current is not None and line.strip():
            fields[current] += " " + line.strip()
    if "Story" in fields:
        fields["Story"] = fields["Story"].strip()
    return fields


def build_request(fields, rng, keep_all=False):
    summary = fields.get("Summary", "")
    words = [w.strip() for w in fields.get("Words", "").split(",") if w.strip()]
    features = [f.strip() for f in fields.get("Features", "").split(",") if f.strip() in FEATURE_PHRASES]

    keep_summary = summary and (keep_all or rng.random() < KEEP_SUMMARY)
    keep_words = words and (keep_all or rng.random() < KEEP_WORDS)
    keep_features = features and (keep_all or rng.random() < KEEP_FEATURES)

    templates = PLOT_TEMPLATES if keep_summary else NO_PLOT_TEMPLATES
    # the longest template when sizing the full request, so the length filter never depends on randomness
    template = max(templates, key=len) if keep_all else rng.choice(templates)
    parts = [template.format(summary) if keep_summary else template]
    if keep_features:
        parts.extend(FEATURE_PHRASES[f] for f in features)
    if keep_words:
        listed = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        parts.append(f"Use the words {listed}.")
    return " ".join(parts)


def convert(text, max_chars, rng):
    """Returns (examples, stats). The last chunk has no closing <|endoftext|> (possibly cut off), so it is skipped."""
    stats = {"entries": 0, "no_story": 0, "too_long": 0, "kept": 0}
    examples = []
    for entry in text.split(END_OF_TEXT)[:-1]:
        stats["entries"] += 1
        fields = parse_entry(entry.translate(ASCII_MAP))
        story = fields.get("Story", "")
        if not story:
            stats["no_story"] += 1
            continue
        # filter on the longest possible request: filtering the randomly shortened one would favour
        # entries whose summary happened to be dropped
        longest = f"User: {build_request(fields, rng, keep_all=True)}\nAssistant: {story}\n{END_OF_TEXT}\n"
        if len(longest) > max_chars:
            stats["too_long"] += 1
            continue
        examples.append(f"User: {build_request(fields, rng)}\nAssistant: {story}\n{END_OF_TEXT}\n")
        stats["kept"] += 1
    return examples, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    # 256 tokens * ~2.69 chars/token (BPE 512 on TinyStories) is ~690 chars; the Java side must still drop
    # or truncate anything that encodes to more than T_max tokens
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8", errors="replace") as f:
        text = f.read()
    examples, stats = convert(text, args.max_chars, random.Random(args.seed))
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(examples)

    lengths = sorted(len(e) for e in examples)
    print(f"{args.input} -> {args.output}")
    print(f"entries {stats['entries']}, kept {stats['kept']} ({100 * stats['kept'] / max(1, stats['entries']):.1f}%), "
          f"too long {stats['too_long']}, no story {stats['no_story']}")
    if lengths:
        print(f"example chars: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
              f"total {sum(lengths) / 1e6:.1f}M chars")


if __name__ == "__main__":
    main()
