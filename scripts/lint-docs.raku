#!/usr/bin/env raku
use v6.d;

=begin pod

=head1 NAME

lint-docs.raku — documentation and diagram conformance gate for libgrammstein.

=head1 SYNOPSIS

    raku scripts/lint-docs.raku                    # scan every doc + diagram source (fast rules)
    raku scripts/lint-docs.raku --render           # additionally render each .puml and scan the SVG
    raku scripts/lint-docs.raku --fix              # repair the auto-fixable kinds in place
    raku scripts/lint-docs.raku --list-rules       # describe every rule
    raku scripts/lint-docs.raku --rules=blocked-macro docs/api/ngram.md

=head1 EXIT STATUS

    0   clean
    1   violations found
    2   usage or environment error

=head1 WHY EACH RULE EXISTS

Every rule here encodes a defect that actually shipped, and that no existing gate caught.

=item B<deprecated-activity-colour> — PlantUML's activity colour-prefix form C<#COLOR:label;> is
deprecated in favour of the suffix form C<:label; E<lt>E<lt>#COLORE<gt>E<gt>>. This is not cosmetic:
PlantUML B<silently drops the colour entirely> and stamps a warning into the rendered image. It exits
0 and writes nothing to stderr, so C<-checkonly> and stderr scanning both see nothing.

=item B<unescaped-underscore-in-text> — inside C<\text{}> an underscore must be written C<\_>. A bare
C<_> is math-mode-only in LaTeX, and GitHub rejects the B<whole span> with
C<'_' allowed only in math mode>. Math-mode macros (C<\mathrm> et al.) are unaffected: there C<\_> is
likewise correct, and a bare C<_> is a legitimate subscript — so they are never flagged.

Beware the trap this rule was originally written backwards to fall into: whether C<\text{a\_b}> is
correct depends on the renderer's B<package list>, not on LaTeX. C<\text> itself is MathJax's C<base>
package, but the parser for its I<contents> is a hook — C<ParseUtil.internalMath> delegates to
C<options.internalMath> only if something installed it, and only the C<textmacros> package does.
Without C<textmacros>, C<\_> is never expanded and renders a B<literal backslash>; with it, C<\_> is an
underscore and a bare C<_> is an error. GitHub loads C<textmacros>. A local previewer that does not is
I<more permissive than the publishing target> and will silently green-light source that GitHub
rejects — measure the renderer, do not infer it.

=item B<tagged-math-block> — C<\tag{X}> makes MathJax emit a B<labelled table> (C<mlabeledtr>) whose
width depends on a container it cannot measure, so GitHub stacks the equation B<vertically>. Rewritten
as C<\begin{array}{lr} \displaystyle … & \text{(X)} \end{array}>, which is an ordinary row. The
C<\displaystyle> is load-bearing: C<array> switches to text style and would silently shrink every
display fraction (measured: 5.47ex → 3.61ex).

=item B<inverted-math-delimiters> — GitHub-flavored Markdown inline math is a backtick span wrapped in
dollars, C<$`x`$>. The inverted C<`$x$`> is an ordinary code span and renders as literal text.

=item B<blocked-macro> — GitHub pre-scans each math span against a macro blocklist and refuses to
render the whole span, printing "The following macros are not allowed: ...". C<\operatorname> is
blocked. Detect-only: the repair is a semantic judgement (C<\mathrm> vs C<\arg\max_{sub}> vs
C<\underset{sub}{...}> vs C<\text{top-}>), not a mechanical substitution.

=item B<render-error>, B<leaked-latex>, B<empty-glyph> — PlantUML renders failures I<into> the image
and still exits 0. C<empty-glyph> decodes each embedded JLaTeXMath image and asserts it drew at least
one glyph; a dropped glyph is still a well-formed C<E<lt>imageE<gt>>, so nothing shallower can see it.

=head1 THE JAVA 21 PIN

The C<--render> rules pin Java 21. This is load-bearing, not incidental: Java 26 regressed the
rendering of the soft-hyphen codepoint U+00AD, which is where C<\otimes> lives in the C<cmsy10> font.
Under Java 26 the C<empty-glyph> rule would fire falsely on every C<\otimes>.

=end pod

# ── Model ────────────────────────────────────────────────────────────────────

#| One finding, reported as `file:line kind: excerpt`.
class Violation {
    has Str  $.file    is required;
    has Int  $.line    is required;
    has Str  $.kind    is required;
    has Str  $.excerpt is required;
    method gist { "$!file:$!line $!kind: $!excerpt" }
}

#| A rule. `fix` is defined only for kinds whose repair is deterministic; the rest are detect-only
#| because auto-fixing them would require judgement the linter does not have.
class Rule {
    has Str      $.name        is required;
    has Str      $.applies     is required;   # 'md' | 'puml' | 'svg'
    has Str      $.description is required;
    has Callable $.detect      is required;   # (Str $file, Str $text --> List of Violation)
    has Callable $.fix;                       # (Str $text --> Str)
    method fixable(--> Bool) { $!fix.defined }
}

# ── Shared helpers ───────────────────────────────────────────────────────────

#| How a markdown line relates to the math rules.
#|   Skip      — inside an opaque fence (```rust …), or the fence marker itself.
#|   MathFence — inside a ```math block: the WHOLE line is display math.
#|   Prose     — an ordinary line: math lives only inside its `$`…`$` inline spans.
enum LineMode < Skip MathFence Prose >;

#| Classify every markdown line as (line-number, text, LineMode).
#|
#| Fenced blocks are opaque — EXCEPT ```math, which *is* display math. That distinction keeps the two
#| inert `\label`s (inside Rust fences) from tripping blocked-macro while still catching a real
#| `\operatorname` inside a ```math block.
#|
#| DETECTORS AND FIXERS MUST BOTH GO THROUGH THIS. Two bugs came from a fixer that did not:
#|   1. A whole-document `subst` pairs backticks ACROSS lines, because `<-[`]>` also matches "\n".
#|      The pairing parity then slips and the fixer "repairs" the gap between two *correct* spans —
#|      turning `$`a`$ and $`b`$` into `$`a$` and `$b`$`, mangling both.
#|   2. A whole-document `subst` is fence-blind and would rewrite math inside a ```rust example.
sub md-lines(Str $text) {
    my $fence-lang;                                     # Nil <=> outside a fence
    gather for $text.lines.kv -> $i, $line {
        if $line ~~ / ^ \h* '```' \h* $<lang>=[\w*] / {
            $fence-lang = $fence-lang.defined ?? Nil !! ~$<lang>;
            take ($i + 1, $line, Skip);                 # the fence marker itself is never scanned
        } elsif !$fence-lang.defined {
            take ($i + 1, $line, Prose);
        } else {
            take ($i + 1, $line, $fence-lang eq 'math' ?? MathFence !! Skip);
        }
    }
}

#| An inline GFM math span: a backtick run wrapped in dollars, `$`…`$`.
#|
#| This is what separates math from a code span that merely TALKS about math. GFM math *is* a backtick
#| span, so "skip code spans" would skip all math; the discriminator is the surrounding dollars.
#| Without it, prose documenting the rules (`` `\text{a\_b}` ``, `` `\operatorname` ``) self-trips —
#| this README and this linter's own docs did exactly that.
my regex math-span { '$' $<open>=[ '`'+ ] $<body>=[ <-[` \n]>* ] $<close>=[ '`'+ ] '$' }

#| The math content of a line, given its mode: whole line in a ```math fence, else each inline span.
sub math-regions(Str $line, LineMode $mode) {
    return ($line,) if $mode == MathFence;
    return () unless $mode == Prose;
    gather for $line ~~ m:g/ <math-span> / -> $m {
        take ~$m<math-span><body> if ~$m<math-span><open> eq ~$m<math-span><close>;
    }
}

#| Rewrite the math content of a line with &fn, leaving code spans and prose untouched.
sub map-math-regions(Str $line, LineMode $mode, &fn --> Str) {
    return fn($line) if $mode == MathFence;
    return $line unless $mode == Prose;
    $line.subst(&math-span, {
        my $whole = $/.Str;
        my $open  = ~$/<open>;
        my $close = ~$/<close>;
        my $body  = ~$/<body>;
        $open eq $close ?? '$' ~ $open ~ fn($body) ~ $close ~ '$' !! $whole;
    }, :g);
}

#| Rebuild a document with &fn applied to the math content of every line.
sub md-map-math(Str $text, &fn --> Str) {
    my $body = md-lines($text).map({ map-math-regions(.[1], .[2], &fn) }).join("\n");
    # `.lines` discards the trailing newline; restore it so --fix never strips the final EOL.
    $text.ends-with("\n") ?? $body ~ "\n" !! $body;
}

#| Rebuild a document with &fn applied to whole Prose lines (for rules about the delimiters
#| themselves, which live outside any math span by definition).
sub md-map-lines(Str $text, &fn --> Str) {
    my $body = md-lines($text).map({ .[2] == Prose ?? fn(.[1]) !! .[1] }).join("\n");
    $text.ends-with("\n") ?? $body ~ "\n" !! $body;
}

#| Every ```math block as (first-body-line-number, body-as-one-string).
#|
#| Some rules are about the block as a WHOLE rather than about a line: a display equation is one
#| expression that happens to be typed across several source lines (TeX ignores the newlines), so a
#| per-line traversal cannot see it. md-map-math / md-map-lines remain correct for the span-level and
#| delimiter-level rules; this is the third shape.
sub math-blocks(Str $text) {
    my $in = False;
    my @body;
    my $start = 0;
    gather for $text.lines.kv -> $i, $line {
        if !$in && $line ~~ / ^ \h* '```math' \h* $ / { $in = True; @body = (); $start = $i + 2; next }
        if $in && $line ~~ / ^ \h* '```' \h* $ /      { $in = False; take ($start, @body.join("\n")); next }
        @body.push($line) if $in;
    }
}

#| Rebuild a document with &fn applied to the body of every ```math block, fences left intact.
sub md-map-math-blocks(Str $text, &fn --> Str) {
    my $in = False;
    my @body;
    my @out;
    for $text.lines -> $line {
        if !$in && $line ~~ / ^ \h* '```math' \h* $ / { $in = True; @body = (); @out.push($line); next }
        if $in && $line ~~ / ^ \h* '```' \h* $ / {
            $in = False;
            @out.push(|fn(@body.join("\n")).lines);
            @out.push($line);
            next;
        }
        $in ?? @body.push($line) !! @out.push($line);
    }
    @out.push(|@body) if $in;                       # unterminated fence: emit verbatim, never eat it
    my $joined = @out.join("\n");
    $text.ends-with("\n") ?? $joined ~ "\n" !! $joined;
}

#| Text-mode font macros. Inside these, `\_` renders a literal backslash under MathJax.
my regex text-macro { '\\text' [ 'tt' | 'bf' | 'it' | 'rm' | 'sf' ]? }

#| A braced body tolerating one level of nesting, e.g. `\text{(if \texttt{a\_b} > 0)}`.
#| Newlines are excluded: a macro body never spans lines, and allowing it to would let the match run
#| past the end of a line hunting for a closing brace.
my regex braced-body { '{' <-[{}\n]>* [ '{' <-[{}\n]>* '}' <-[{}\n]>* ]* '}' }

#| A CommonMark code span: matching backtick runs with no backtick between them.
#|
#| Excluding "\n" from the body is load-bearing. `<-[`]>` matches newlines, so over a whole document
#| the engine happily pairs a backtick on one line with one several lines later; the pairing parity
#| then slips and `$`a`$ and $`b`$` gets "repaired" into `$`a$` and `$b`$`. Both this regex and the
#| line-scoped md-map traversal exist to make that impossible.
my regex code-span { $<open>=[ '`'+ ] $<body>=[ <-[` \n]>+ ] $<close>=[ '`'+ ] }

#| Macros GitHub refuses to render. Raku's longest-token-match makes `\operatorname` win over `\op…`,
#| and the trailing `>>` word boundary stops `\include` from matching `\includegraphics`.
my constant @BLOCKED-MACROS = <
    operatorname DeclareMathOperator newcommand renewcommand providecommand
    newenvironment renewenvironment def edef gdef xdef let require input include
    href unicode class cssId style bbox htmlClass htmlId htmlData htmlStyle
>;

#| Pure-Raku base64 decoder — deliberately dependency-free so the gate needs nothing but rakudo.
my constant @B64-ALPHABET = |('A'..'Z'), |('a'..'z'), |('0'..'9'), '+', '/';
my constant %B64-INDEX    = @B64-ALPHABET.antipairs.Hash;
sub b64-decode(Str $data --> Str) {
    my ($acc, $nbits) = 0, 0;
    my @bytes;
    for $data.comb -> $c {
        my $v = %B64-INDEX{$c};
        next without $v;
        $acc = ($acc +< 6) +| $v;
        $nbits += 6;
        if $nbits >= 8 {
            $nbits -= 8;
            @bytes.push: ($acc +> $nbits) +& 0xFF;
        }
    }
    Buf.new(@bytes).decode('utf-8', :replacement(''));
}

# ── PlantUML invocation (see "THE JAVA 21 PIN" above) ────────────────────────

my constant $JAVA21 = '/usr/lib/jvm/java-21-openjdk/bin/java';
my constant $PUML-JAR = '/usr/share/java/plantuml/plantuml.jar';

#| Render one .puml to SVG text via pipe mode, so nothing is ever written to the repo.
sub render-puml(IO::Path $src --> Str) {
    my $proc = run $JAVA21, '-jar', $PUML-JAR, '-tsvg', '-p', :in, :out, :err;
    $proc.in.print: $src.slurp;
    $proc.in.close;
    my $svg = $proc.out.slurp(:close);
    $proc.err.slurp(:close);
    $svg;
}

#| Decode XML character entities AND fold the non-breaking-space family to U+0020, so diagnostics can
#| be matched as ordinary prose.
#|
#| Both halves are load-bearing, and each hid a false PASS during development:
#|   1. PlantUML emits label text with entities — the warning is stored as
#|      `This&#160;syntax&#160;is&#160;deprecated,&#160;…`, so searching the raw SVG finds nothing.
#|   2. `&#160;` decodes to U+00A0 (NBSP), *not* U+0020 — so even after decoding, a literal search
#|      written with ordinary spaces still fails. The words must be re-joined by real spaces.
#| `&amp;` is decoded last so an encoded `&amp;#160;` cannot be mistaken for a space.
sub decode-entities(Str $svg --> Str) {
    $svg.subst(/ '&#' $<n>=[\d+] ';' /, { chr(+$<n>) }, :g)
        .subst('&nbsp;', ' ', :g)
        .subst('&lt;',   '<', :g)
        .subst('&gt;',   '>', :g)
        .subst('&quot;', '"', :g)
        .subst('&amp;',  '&', :g)
        .subst(/ <[ \x[00A0] \x[2007] \x[202F] ]> /, ' ', :g);
}

# ── Rules ────────────────────────────────────────────────────────────────────

#| `#COLOR:label…;` -> `:label…; <<#COLOR>>`.
#|
#| The colour prefix opens the activity but the `;` may close it several lines later, so the token
#| moves across lines. The terminator is anchored to end-of-line (`';' \h* $$`) — without that anchor
#| a mid-line `;` (e.g. the Rust type `&str;` in a label) would be mistaken for the terminator and
#| silently truncate the label.
my regex deprecated-activity { ^^ (\h*) '#' $<col>=[ <[0..9A..Fa..f]> ** 6 ] ':' $<body>=[ .*? ] ';' \h* $$ }

sub fix-activity-colour(Str $text --> Str) {
    $text.subst(&deprecated-activity, { $0 ~ ':' ~ $<body> ~ '; <<#' ~ $<col> ~ '>>' }, :g, :s);
}

my $rule-activity = Rule.new(
    name        => 'deprecated-activity-colour',
    applies     => 'puml',
    description => 'PlantUML activity colour prefix `#COLOR:label;` — deprecated; the colour is silently dropped. Use `:label; <<#COLOR>>`.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        for $text.lines.kv -> $i, $line {
            @v.push: Violation.new(
                :$file, line => $i + 1, kind => 'deprecated-activity-colour',
                excerpt => $line.trim.substr(0, 72),
            ) if $line ~~ / ^^ \h* '#' <[0..9A..Fa..f]> ** 6 ':' /;
        }
        @v;
    },
    fix         => &fix-activity-colour,
);

#| A bare underscore — one NOT already escaped. The negative lookbehind is what makes the fixer
#| idempotent: without it, a second pass would turn `\_` into `\\_`.
my regex bare-underscore { <!after '\\'> '_' }

#| `\text{a_b}` -> `\text{a\_b}`, rewriting ONLY inside the text-macro's own braces so that a
#| `\mathrm{a_b}` sharing the region is left untouched (there a bare `_` is a legitimate subscript).
sub fix-underscore-region(Str $math --> Str) {
    $math.subst(/ <text-macro> <braced-body> /,
                { $/.Str.subst(&bare-underscore, '\\_', :g) }, :g);
}

my $rule-underscore = Rule.new(
    name        => 'unescaped-underscore-in-text',
    applies     => 'md',
    description => 'Bare `_` inside a text-mode macro (\text/\texttt/…). `_` is math-mode-only in LaTeX, so GitHub rejects the whole span with "\'_\' allowed only in math mode". Write `\_`. Math-mode macros (\mathrm) are unaffected and never flagged.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        for md-lines($text) -> ($lno, $line, $mode) {
            for math-regions($line, $mode) -> $math {
                for $math ~~ m:g/ <text-macro> <braced-body> / -> $m {
                    @v.push: Violation.new(
                        :$file, line => $lno, kind => 'unescaped-underscore-in-text',
                        excerpt => $m.Str.substr(0, 72),
                    ) if $m.Str ~~ &bare-underscore;
                }
            }
        }
        @v;
    },
    fix         => -> Str $text { md-map-math($text, &fix-underscore-region) },
);

#| `` `$x$` `` -> `` $`x`$ ``. Requiring the code-span content to BOTH start and end with `$` is what
#| keeps currency (`$5`), shell vars (`$HOME`, `$FOO$BAR`), regex anchors (`$`) and awk (`$1 == $2`)
#| clean without any allowlist.
sub inverted-hits(Str $line) {
    my @hits;
    for $line ~~ m:g/ <code-span> / -> $m {
        next unless ~$m<code-span><open> eq ~$m<code-span><close>;
        @hits.push: $m if ~$m<code-span><body> ~~ / ^ '$' .* '$' $ /;
    }
    @hits;
}

sub fix-inverted-delimiters(Str $text --> Str) {
    md-map-lines($text, -> $line {
        $line.subst(
            &code-span,
            {
                # Bind every capture to a lexical BEFORE the guard below runs. The `~~` inside the
                # guard rebinds $/ to its own match, so reading $/<body> afterwards yields Nil and
                # silently replaces the span with an empty one — i.e. it eats the expression.
                my $whole = $/.Str;
                my $open  = ~$/<open>;
                my $close = ~$/<close>;
                my $body  = ~$/<body>;
                ($open eq $close && $body ~~ / ^ '$' .* '$' $ /)
                    # The body cannot contain a backtick (matched by <-[`]>), so one tick is safe.
                    ?? '$`' ~ $body.substr(1, *-1) ~ '`$'
                    !! $whole;
            },
            :g,
        )
    });
}

my $rule-inverted = Rule.new(
    name        => 'inverted-math-delimiters',
    applies     => 'md',
    description => 'Inverted GFM math delimiters: `` `$x$` `` is a code span, not math. Dollars go OUTSIDE the backticks: `` $`x`$ ``.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        for md-lines($text).grep({ .[2] == Prose }) -> ($lno, $line, $) {
            for inverted-hits($line) -> $m {
                @v.push: Violation.new(
                    :$file, line => $lno, kind => 'inverted-math-delimiters',
                    excerpt => $m.Str.substr(0, 72),
                );
            }
        }
        @v;
    },
    fix         => &fix-inverted-delimiters,
);

my $rule-blocked = Rule.new(
    name        => 'blocked-macro',
    applies     => 'md',
    description => 'A macro GitHub refuses to render in math spans (\operatorname, \newcommand, \require, …). Detect-only: the repair is a semantic judgement.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        # Scoped to math regions: a `\operatorname` named in prose or shown inside a code span is
        # documentation, not a broken math span.
        for md-lines($text) -> ($lno, $line, $mode) {
            for math-regions($line, $mode) -> $math {
                for $math ~~ m:g/ '\\' $<macro>=[ @BLOCKED-MACROS ] >> / -> $m {
                    @v.push: Violation.new(
                        :$file, line => $lno, kind => 'blocked-macro',
                        excerpt => "\\{$m<macro>} — not renderable on GitHub",
                    );
                }
            }
        }
        @v;
    },
);

my $rule-abutting = Rule.new(
    name        => 'delimiter-abutting',
    applies     => 'md',
    description => 'An ASCII letter abutting an opening `` $` `` delimiter; GitHub will not open the math span.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        # Anchored on a REAL math span, not on the two characters `$` and a backtick. Prose that
        # merely shows the inverted form — `` `$x$` `` — has no math span and must not be flagged;
        # matching the raw characters made this linter's own README self-trip.
        for md-lines($text).grep({ .[2] == Prose }) -> ($lno, $line, $) {
            for $line ~~ m:g/ <[A..Za..z]> <math-span> / -> $m {
                @v.push: Violation.new(
                    :$file, line => $lno, kind => 'delimiter-abutting',
                    excerpt => $m.Str.substr(0, 40),
                );
            }
        }
        @v;
    },
);

#| Rules over RENDERED output. PlantUML exits 0 and reports failures inside the image, so these can
#| only run after a render.
#|
#| The deprecation matcher uses PlantUML's exact sentence rather than the bare word "deprecated":
#| ngram-trie-storage.puml legitimately *says* "deprecated" in its own labels (a legend row
#| `| Deprecated / unsafe |`, and `**Legacy encoding** (deprecated)`) while containing no deprecated
#| syntax. A case-insensitive word match would flag it forever.
my $rule-render = Rule.new(
    name        => 'render-diagnostics',
    applies     => 'svg',
    description => 'Rendered-SVG diagnostics: deprecated-syntax, render-error, leaked-latex, empty-glyph.',
    detect      => sub (Str $file, Str $svg) {
        my @v;
        my sub note(Str $kind, Str $excerpt) {
            @v.push: Violation.new(:$file, line => 0, :$kind, :$excerpt);
        }

        # Entity-decoded, because PlantUML stores label text with &#160; between words.
        my $prose = decode-entities($svg);

        # The rendered symptom; the source-level cause is `deprecated-activity-colour`, which --fix
        # repairs. Say so, because this kind itself carries no fixer.
        if $prose ~~ / 'This syntax is deprecated' <-[<]>* / {
            note('deprecated-syntax',
                 ~$/.trim.subst(/\s+/, ' ', :g).substr(0, 76) ~ ' — repair the source with --fix');
        }

        for 'Syntax Error?', 'Unknown symbol', 'cannot find' -> $marker {
            note('render-error', "PlantUML error graphic: $marker") if $prose.contains($marker);
        }

        note('leaked-latex', 'unrendered <latex> literal survived into the SVG')
            if $prose.contains('<latex>');

        # Each JLaTeXMath expression is embedded as a nested SVG data-URI. A dropped glyph still
        # yields a well-formed <image>, so counting images proves nothing — decode and look inside.
        my $idx = 0;
        for $svg ~~ m:g/ 'base64,' $<data>=[ <[A..Za..z0..9+/=]>+ ] / -> $m {
            $idx++;
            my $inner = b64-decode(~$m<data>);
            next unless $inner.contains('<svg');           # only inspect nested SVG images
            note('empty-glyph', "embedded LaTeX image #$idx decoded to 0 glyph paths")
                unless $inner.contains('<path');
        }
        @v;
    },
);

#| `X \tag{M1}` -> an array row: the label becomes an ordinary right-hand cell.
#|
#| `\displaystyle` is REQUIRED, not decoration: `\begin{array}` switches to TEXT style, which silently
#| shrinks display fractions (measured: height 5.47ex -> 3.61ex). With it, the height matches the
#| untagged baseline exactly.
#|
#| Two facts from surveying all 462 tagged blocks make this a single uniform rewrite: `\tag` is the
#| last token in every one of them, and none has a top-level `\\` (so no `gathered` wrapper is ever
#| needed — a bare `\\` would otherwise start a new array ROW and scramble the layout).
my regex trailing-tag { '\tag' \h* '{' $<label>=[ <-[}]>* ] '}' \s* $ }

sub fix-tagged-block(Str $body --> Str) {
    return $body unless $body ~~ &trailing-tag;
    my $label = ~$/<label>;
    my $math  = $body.subst(&trailing-tag, '').trim-trailing;
    "\\begin\{array\}\{lr\}\n\\displaystyle $math & \\text\{($label)\}\n\\end\{array\}";
}

my $rule-tagged = Rule.new(
    name        => 'tagged-math-block',
    applies     => 'md',
    description => '`\tag{X}` in a ```math block. MathJax renders it as a labelled table whose width depends on a container it does not have, so GitHub stacks the equation VERTICALLY. Use `\begin{array}{lr} \displaystyle … & \text{(X)} \end{array}`.',
    detect      => sub (Str $file, Str $text) {
        my @v;
        for math-blocks($text) -> ($lno, $body) {
            next unless $body ~~ / '\tag' \h* '{' $<label>=[ <-[}]>* ] '}' /;
            @v.push: Violation.new(
                :$file, line => $lno, kind => 'tagged-math-block',
                excerpt => "\\tag\{{$<label>}\} — renders vertically on GitHub",
            );
        }
        @v;
    },
    fix         => -> Str $text { md-map-math-blocks($text, &fix-tagged-block) },
);

my @ALL-RULES = $rule-activity, $rule-underscore, $rule-inverted, $rule-blocked, $rule-abutting,
                $rule-tagged, $rule-render;

# ── Driver ───────────────────────────────────────────────────────────────────

#| Recursively list files under $d. Relies on `.dir`'s default test, which excludes `.` and `..` —
#| overriding it with `:test(*)` makes the walk ascend into the parent and re-scan everything.
sub slip-files(IO::Path $d) {
    $d.dir.map({ .d ?? slip($_.&slip-files) !! $_ }).flat;
}

sub default-targets {
    my @t;
    for 'docs', 'formal' -> $dir {
        @t.append: slip-files($dir.IO) if $dir.IO.d;
    }
    @t.push: 'README.md'.IO if 'README.md'.IO.e;
    @t.grep({ .extension eq 'md' | 'puml' }).unique(:as(*.absolute));
}

sub kind-of(IO::Path $p) { $p.extension eq 'puml' ?? 'puml' !! 'md' }

sub MAIN(
    *@paths,
    Bool :$fix,                 #= repair the auto-fixable kinds in place
    Bool :$render,              #= also render each .puml and scan the SVG (slow; needs java 21 + plantuml)
    Bool :$list-rules,          #= describe every rule and exit
    Str  :$rules,               #= comma-separated rule names to run (default: all)
) {
    if $list-rules {
        say 'Rules:';
        for @ALL-RULES -> $r {
            say sprintf('  %-28s [%-4s] %s%s', $r.name, $r.applies,
                        $r.fixable ?? '(auto-fixable) ' !! '(detect-only)  ', $r.description);
        }
        exit 0;
    }

    my @selected = @ALL-RULES;
    if $rules {
        my %want = $rules.split(',')».trim.Set;
        # 'render-diagnostics' owns four kinds; let callers name any of them.
        @selected = @ALL-RULES.grep({
            %want{.name} || (.name eq 'render-diagnostics'
                             && %want{any <deprecated-syntax render-error leaked-latex empty-glyph>})
        });
        unless @selected {
            note "error: no rules matched '$rules'. Try --list-rules.";
            exit 2;
        }
    }

    my @targets = @paths
        ?? @paths.map(*.IO).map({ .d ?? slip($_.&slip-files) !! $_ }).flat
                 .grep({ .extension eq 'md' | 'puml' }).unique(:as(*.absolute))
        !! default-targets();
    unless @targets {
        note 'error: no .md or .puml files to lint';
        exit 2;
    }

    if $render {
        for $JAVA21, $PUML-JAR -> $needed {
            unless $needed.IO.e {
                note "error: --render needs $needed (java 21 is pinned: java 26 regressed the U+00AD glyph, which would make empty-glyph fire falsely on every \\otimes)";
                exit 2;
            }
        }
    }

    my @violations;
    my @fixed;

    for @targets.sort -> $path {
        my $kind = kind-of($path);
        my $text = $path.slurp;
        my $orig = $text;

        # Repair first, then report against the FINAL text. Detecting before and after would
        # double-count every detect-only kind.
        if $fix {
            for @selected.grep({ .applies eq $kind && .fixable }) -> $rule {
                $text = $rule.fix.($text);
            }
            if $text ne $orig {
                $path.spurt($text);
                @fixed.push: ~$path;
            }
        }

        # Whatever survives is a real violation: in --fix mode the fixable kinds should now be 0,
        # and the detect-only kinds remain reported for a human.
        for @selected.grep({ .applies eq $kind }) -> $rule {
            @violations.append: $rule.detect.(~$path, $text);
        }
    }

    if $render {
        my @pumls = @targets.grep({ .extension eq 'puml' }).sort;
        note "lint-docs: rendering {+@pumls} diagram(s) under java 21…";
        for @pumls -> $p {
            @violations.append: $rule-render.detect.(~$p, render-puml($p));
        }
    }

    if $fix {
        if @fixed {
            say "lint-docs: repaired {+@fixed} file(s):";
            say "  $_" for @fixed;
        } else {
            say 'lint-docs: nothing to repair.';
        }
    }

    unless @violations {
        say "lint-docs: ✅ PASS — 0 violations across {+@targets} file(s)"
            ~ ($render ?? ' (including rendered-SVG diagnostics).' !! '.');
        exit 0;
    }

    say .gist for @violations.sort({ .file, .line });
    say '──────────────────────────────────────────────────────────────────────────────';
    my %by-kind;
    %by-kind{.kind}++ for @violations;
    my %fixable-kind = @ALL-RULES.grep(*.fixable).map({ .name => True });
    say "lint-docs: ❌ FAIL — {+@violations} violation(s).";
    say '   Kinds:';
    for %by-kind.sort(-*.value) -> $k {
        say sprintf('     %4d  %-28s %s', $k.value, $k.key,
                    %fixable-kind{$k.key} ?? '(--fix can repair this)' !! '(needs a human)');
    }
    exit 1;
}
