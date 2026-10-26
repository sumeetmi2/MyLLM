# Contributing

How code, comments and documentation are written in this repo, and what the checks enforce.
Setup for a fresh clone:

```
./gradlew installGitHooks   # once: git runs .githooks/pre-commit before every commit
./gradlew build             # compile + lint + tests
```

## Comments

The rules come from Google Java Style (sections 4.8.6 and 7), Oracle's "How to Write Doc Comments for the Javadoc
Tool" and Effective Java (item 56), cut down to what fits a from-scratch learning codebase.

1. **Say why, not what.** The code already says what it does. A comment earns its place by giving the reason, the
   math, or the trap: `// rows added in order so the total does not depend on thread scheduling`, not
   `// loop over rows`.
2. **Write the math down.** For a forward or backward pass, name the formula and the shapes it works on
   (`// dX = dY · Wᵀ, (T×d)`). Use the same symbols as `SPEC.md`, so names like `d_Model`, `T_max` and `W_q`
   are allowed on purpose.
3. **Comment the non-obvious numbers.** Every tolerance, epsilon, step size or threshold says where it comes from
   (`// float has ~7 digits, so a 1e-5 step would drown in rounding`).
4. **Keep comments next to what they explain and update them in the same change.** A stale comment is worse than
   none. Reviewers read the comment diff together with the code diff.
5. **No commented-out code, no change logs, no author names.** Git history keeps all three.
6. **`TODO` says what is left and why:** `// TODO: KV cache - generation recomputes the whole prefix per token`.
   Checkstyle (`TodoComment`) rejects a bare `TODO` or `FIXME`.
7. **Implementation comments use `//`.** No boxes or banners made of `*`, `-`, `=` or `#` (Checkstyle rejects
   them). What a class is for goes in its Javadoc, not in a `//` block above it.

## Javadoc

Javadoc (`/** ... */`) is for what a caller needs to know; `//` comments are for whoever edits the body.

- **Every public top-level type has class Javadoc** (`MissingJavadocType`). Nested result types and test
  classes are exempt (`config/checkstyle/suppressions*.xml`).
- **Every package has a `package-info.java`** whose first sentence says what the package holds. Checkstyle
  (`MissingJavadocPackage`) requires it, and the code map in `README.md` is generated from these sentences.
- **Public classes and methods get Javadoc when the name and types don't tell the whole story:** units, shapes,
  ranges, what `null` means, what is thrown and when, side effects on cached state. Obvious members (`getGamma()`)
  and overrides need none. Missing Javadoc is not a build error.
- **The first sentence is the summary.** The javadoc tool copies it into package and class indexes, so it must
  stand alone. Write a verb or noun phrase that ends with a period: `Returns the row-major float buffer.`,
  not `This method returns ...` or `@return the ...` (Checkstyle `SummaryJavadoc` / `JavadocStyle`).
- **Tags** go in the order `@param`, `@return`, `@throws`, `@deprecated`, never with an empty description
  (`AtclauseOrder`, `NonEmptyAtclauseDescription`). Every `@param` names a real parameter (`JavadocMethod`).
- **Paragraphs** after the first start with `<p>` right before the first word (`JavadocParagraph`).
- **Code examples** use `{@snippet}` (JEP 413, JDK 18+) instead of `<pre>{@code ...}</pre>`.
- `./gradlew javadoc` runs doclint on all of it, so a broken `{@link}` or bad HTML fails `lint`.

## Security checks

Only tools with a long track record and low noise, run on every `lint`:

- **Find Security Bugs** (an OWASP project, a SpotBugs plugin): data-flow detectors for command, SQL and
  path injection, unsafe deserialization, weak hashing and ciphers, predictable randomness, XSS, SSRF and more.
  SpotBugs is limited to this SECURITY category; general bug patterns are Error Prone's job.
- **Error Prone** `BanSerializableRead`, `BanJNDI`, `BanClassLoader`: the classic remote-code-execution entry
  points, banned at compile time.

A finding is fixed, or accepted with a reason: in `config/spotbugs/exclude.xml` for SpotBugs (scoped to the
class and method), or `@SuppressWarnings("<check>")` plus a comment for Error Prone. `java.util.Random` is
accepted in the model code because training must be reproducible; use `SecureRandom` for anything secret.

Not included: OWASP Dependency-Check (the app has no runtime dependencies, and it needs an NVD API key), and
Semgrep or CodeQL (they need an external service or CLI outside Gradle).

## Project documentation

The docs follow the [Diátaxis](https://diataxis.fr/) split: each document has one job.

| document | kind | contents |
|---|---|---|
| `README.md` | how-to + reference | running every command; command usage and code map (generated) |
| `SPEC.md` | explanation | architecture and math of the model |
| `CONTRIBUTING.md` | how-to | this file: conventions and checks |
| package-info / Javadoc | reference | per package and per class contract |

**Generated sections.** Parts of `README.md` sit between `BEGIN GENERATED` and `END GENERATED` markers and are
written from the code by `./gradlew updateDocs`:

- `usage`: the output of `Main help`, so a new command or option shows up in the README as soon as `usage()`
  in `Main.java` documents it
- `packages`: the first sentence of each `package-info.java`

Never edit inside the markers; edit the source (the `usage()` text block, or the `package-info.java`) instead.

## What runs when

| when | what | on failure |
|---|---|---|
| `git commit` (pre-commit hook) | `updateDocs`, then `lint` | README regenerated: staged automatically when you had no unstaged README edits, otherwise the commit stops so you can review. Lint failure stops the commit |
| `./gradlew lint` | javac `-Xlint:all`, Error Prone, Checkstyle (code + comment rules), Spotless format check, javadoc doclint, `checkDocs`, SpotBugs security | fails |
| `./gradlew build` | everything in `lint` plus the tests | fails |

`checkDocs` fails when a generated section is stale, or when a `*.md` file links to a file that doesn't exist or
mentions a repo path in backticks (`src/...`, `config/...`, `scripts/...`, `gradle/...`, `docs/...`,
`.githooks/...`) that doesn't exist. Renaming a file without updating the docs is caught this way.

The hook checks the working tree, not only the staged files. Skip it once with `git commit --no-verify`.
