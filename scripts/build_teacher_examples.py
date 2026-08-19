import html
import json
import pathlib

data = json.loads(
    pathlib.Path("/Users/akshayballal/.claude/jobs/c7968b33/tmp/examples.json").read_text()
)

# Scored on 12 passages x 3 samples, negation prompt, occurrence-counted metric.
SCORES = {
    "Gemini 3.7 Flash": dict(ratio=0.25, vr=0.001, ag=0.004, neg=0.82, num=0.74, order=0.19),
    "Sonnet 4.6": dict(ratio=0.47, vr=0.004, ag=0.082, neg=0.74, num=0.90, order=0.39),
    "Haiku 4.5": dict(ratio=0.54, vr=0.009, ag=0.078, neg=0.76, num=0.91, order=0.44),
    "Kimi K2.5": dict(ratio=0.65, vr=0.001, ag=0.045, neg=0.79, num=0.91, order=0.27),
}
SLUG = {
    "Gemini 3.7 Flash": "gemini",
    "Sonnet 4.6": "sonnet",
    "Haiku 4.5": "haiku",
    "Kimi K2.5": "kimi",
}
TOTALS = {t["name"]: t for t in data["totals"]}
CAPTION = {
    9: "A hiring resolution and roll-call vote. Watch the closing line.",
    2: "A short public-comment item, dense with figures.",
    1: "A procedural item — the least room to cut without losing content.",
}


def markup(words, labels):
    out = []
    for w, keep in zip(words, labels):
        cls = "k" if keep else "d"
        out.append(f'<span class="{cls}">{html.escape(w)}</span>')
    return " ".join(out)


panels = []
for p in data["passages"]:
    src_words = p["words"]
    blocks = []
    for m in p["models"]:
        slug = SLUG[m["name"]]
        neg = "—" if m["negation"] is None else f"{m['negation']:.2f}"
        negcls = " warn" if (m["negation"] is not None and m["negation"] < 1) else ""
        notes = []
        if m["envelope"]:
            notes.append(
                f'<span class="flag">added a title</span> {html.escape(m["envelope"])}'
            )
        if m["novel"]:
            joined = ", ".join(html.escape(w) for w in m["novel"])
            notes.append(f'<span class="flag">not in source</span> {joined}')
        note_html = (
            f'<div class="notes">{"".join(f"<p>{n}</p>" for n in notes)}</div>' if notes else ""
        )
        blocks.append(f"""
        <article class="panel" data-m="{slug}">
          <header class="phead">
            <h4>{html.escape(m["name"])}</h4>
            <dl class="mini">
              <div><dt>kept</dt><dd>{sum(m["labels"])}<span class="of">/{len(src_words)}</span></dd></div>
              <div><dt>ratio</dt><dd>{m["ratio"]:.2f}</dd></div>
              <div><dt>invented</dt><dd class="{"warn" if m["novel_count"] else ""}">{m["novel_count"]}</dd></div>
              <div><dt>negations</dt><dd class="{negcls.strip()}">{neg}</dd></div>
            </dl>
          </header>
          <div class="text mark">{markup(src_words, m["labels"])}</div>
          <div class="text plain">{html.escape(m["compressed"])}</div>
          {note_html}
        </article>""")

    panels.append(f"""
      <section class="passage">
        <div class="phdr">
          <h3>Passage {p["index"]}</h3>
          <p>{CAPTION.get(p["index"], "")} <span class="wc">{len(src_words)} words</span></p>
        </div>
        <details class="src">
          <summary>Read the source transcript</summary>
          <div class="text">{html.escape(" ".join(src_words))}</div>
        </details>
        <div class="grid">{"".join(blocks)}</div>
      </section>""")

rows = []
for name in ["Gemini 3.7 Flash", "Sonnet 4.6", "Haiku 4.5", "Kimi K2.5"]:
    s = SCORES[name]
    t = TOTALS[name]
    rows.append(f"""
      <tr data-m="{SLUG[name]}">
        <th scope="row"><span class="swatch"></span>{html.escape(name)}</th>
        <td>{s["ratio"]:.2f}</td><td>{s["vr"]:.3f}</td><td>{s["ag"]:+.3f}</td>
        <td>{s["neg"]:.2f}</td><td>{s["num"]:.2f}</td><td>{s["order"]:.2f}</td>
        <td class="{"warn" if t["novel"] > 20 else ""}">{t["novel"]}</td>
      </tr>""")

page = f"""<title>Four Teachers, One Transcript</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --paper:#F6F6F4; --raised:#FFFFFF; --ink:#16191C; --ink-2:#4A5056; --ink-3:#767C82;
  --drop:#B9BDBA; --line:#E2E2DE; --line-2:#CFCFC9; --pencil:#B03A28;
  --gemini:#2E6F5E; --sonnet:#8A4B7D; --haiku:#A9531F; --kimi:#2C5AA0;
  --shadow:0 1px 2px rgba(20,22,26,.05), 0 8px 24px -16px rgba(20,22,26,.28);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#131619; --raised:#1A1E22; --ink:#E9E7E2; --ink-2:#AEB4B9; --ink-3:#818890;
    --drop:#4B5157; --line:#272C31; --line-2:#353B41; --pencil:#E4795F;
    --gemini:#63AE93; --sonnet:#C88CBC; --haiku:#DE9059; --kimi:#7BA6DE;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -18px rgba(0,0,0,.8);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#131619; --raised:#1A1E22; --ink:#E9E7E2; --ink-2:#AEB4B9; --ink-3:#818890;
  --drop:#4B5157; --line:#272C31; --line-2:#353B41; --pencil:#E4795F;
  --gemini:#63AE93; --sonnet:#C88CBC; --haiku:#DE9059; --kimi:#7BA6DE;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -18px rgba(0,0,0,.8);
}}
*,*::before,*::after {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1120px; margin:0 auto; padding:clamp(2rem,5vw,4.5rem) clamp(1rem,4vw,2.5rem) 5rem; }}
h1,h2,h3,h4 {{ font-family:Newsreader,Georgia,serif; font-weight:500; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(2.1rem,5.2vw,3.4rem); line-height:1.06; letter-spacing:-.02em; }}
.eyebrow {{
  font-size:.7rem; text-transform:uppercase; letter-spacing:.15em;
  color:var(--ink-3); font-weight:600; margin:0 0 1.1rem;
}}
.lede {{ font-family:Newsreader,Georgia,serif; font-size:1.24rem; line-height:1.55;
  color:var(--ink-2); max-width:60ch; margin:1.4rem 0 0; }}
.lede em {{ color:var(--ink); font-style:italic; }}
header.top {{ border-bottom:1px solid var(--line); padding-bottom:2.5rem; margin-bottom:2.5rem; }}

.tablewrap {{ overflow-x:auto; margin:0 0 1rem; }}
table {{ border-collapse:collapse; width:100%; min-width:660px; font-variant-numeric:tabular-nums; }}
caption {{ text-align:left; color:var(--ink-3); font-size:.82rem; padding-bottom:.7rem; }}
th,td {{ padding:.62rem .7rem; text-align:right; border-bottom:1px solid var(--line); }}
thead th {{
  font-size:.68rem; text-transform:uppercase; letter-spacing:.09em;
  color:var(--ink-3); font-weight:600; border-bottom:1px solid var(--line-2);
}}
tbody th {{ text-align:left; font-weight:500; white-space:nowrap; }}
td {{ font-family:"IBM Plex Mono",monospace; font-size:.86rem; }}
.swatch {{
  display:inline-block; width:9px; height:9px; border-radius:2px;
  background:var(--accent); margin-right:.6rem; vertical-align:baseline;
}}
[data-m="gemini"] {{ --accent:var(--gemini); }}
[data-m="sonnet"] {{ --accent:var(--sonnet); }}
[data-m="haiku"]  {{ --accent:var(--haiku); }}
[data-m="kimi"]   {{ --accent:var(--kimi); }}
.warn {{ color:var(--pencil); font-weight:500; }}

.legend {{
  display:flex; flex-wrap:wrap; gap:1.4rem; align-items:center;
  padding:1rem 1.15rem; background:var(--raised); border:1px solid var(--line);
  border-radius:3px; margin:2.6rem 0 0; font-size:.84rem; color:var(--ink-2);
}}
.legend b {{ font-weight:500; color:var(--ink); }}
.sample-d {{ color:var(--drop); text-decoration:line-through; text-decoration-thickness:1px; }}
.toggle {{
  margin-left:auto; display:inline-flex; align-items:center; gap:.55rem;
  font-size:.8rem; color:var(--ink-2); cursor:pointer; user-select:none;
}}
.toggle input {{ accent-color:var(--ink); width:1rem; height:1rem; cursor:pointer; }}

.passage {{ margin-top:3.6rem; }}
.phdr {{ display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap;
  border-bottom:1px solid var(--line-2); padding-bottom:.7rem; }}
.phdr h3 {{ font-size:1.5rem; }}
.phdr p {{ margin:0; color:var(--ink-3); font-size:.86rem; }}
.wc {{ font-family:"IBM Plex Mono",monospace; font-size:.78rem; }}

details.src {{ margin-top:1rem; }}
details.src summary {{
  cursor:pointer; font-size:.8rem; color:var(--ink-3);
  text-transform:uppercase; letter-spacing:.09em; font-weight:600; padding:.35rem 0;
}}
details.src summary:hover {{ color:var(--ink); }}
details.src .text {{ margin-top:.7rem; padding:1rem 1.15rem; background:var(--raised);
  border:1px solid var(--line); border-radius:3px; color:var(--ink-2); }}

.grid {{ display:grid; gap:1.15rem; margin-top:1.5rem;
  grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }}
.panel {{
  background:var(--raised); border:1px solid var(--line); border-top:2px solid var(--accent);
  border-radius:3px; padding:1.15rem 1.25rem 1.25rem; box-shadow:var(--shadow);
  display:flex; flex-direction:column; gap:.9rem;
}}
.phead {{ display:flex; flex-direction:column; gap:.7rem; }}
.phead h4 {{ font-size:1.16rem; color:var(--accent); }}
dl.mini {{ display:flex; flex-wrap:wrap; gap:1.25rem; margin:0; }}
dl.mini div {{ display:flex; flex-direction:column; gap:.1rem; }}
dl.mini dt {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.1em; color:var(--ink-3); font-weight:600; }}
dl.mini dd {{ margin:0; font-family:"IBM Plex Mono",monospace; font-size:.94rem; font-variant-numeric:tabular-nums; }}
dl.mini .of {{ color:var(--ink-3); font-size:.74rem; }}

.text {{ font-family:Newsreader,Georgia,serif; font-size:1.01rem; line-height:1.68; }}
.mark .k {{ color:var(--ink); }}
.mark .d {{ color:var(--drop); }}
body:not(.show-drops) .mark .d {{ display:none; }}
body.show-drops .mark .d {{ text-decoration:line-through; text-decoration-thickness:1px; }}
body.show-drops .plain {{ display:none; }}
body:not(.show-drops) .mark {{ display:none; }}

.notes {{ border-top:1px solid var(--line); padding-top:.8rem; display:flex; flex-direction:column; gap:.4rem; }}
.notes p {{ margin:0; font-size:.82rem; color:var(--ink-2); font-family:"IBM Plex Mono",monospace; }}
.flag {{
  display:inline-block; font-family:"IBM Plex Sans",sans-serif; font-size:.63rem;
  text-transform:uppercase; letter-spacing:.08em; font-weight:600;
  color:var(--pencil); border:1px solid var(--pencil); border-radius:2px;
  padding:.05rem .34rem; margin-right:.45rem;
}}

.finding {{
  margin-top:3.4rem; padding:1.4rem 1.6rem; background:var(--raised);
  border:1px solid var(--line); border-left:2px solid var(--pencil); border-radius:3px;
}}
.finding h2 {{ font-size:1.3rem; margin-bottom:.6rem; }}
.finding p {{ margin:.55rem 0 0; color:var(--ink-2); max-width:68ch; }}
.finding code {{ font-family:"IBM Plex Mono",monospace; font-size:.86em;
  background:var(--paper); border:1px solid var(--line); border-radius:2px; padding:.08em .32em; }}
footer {{ margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--line);
  color:var(--ink-3); font-size:.8rem; }}
:focus-visible {{ outline:2px solid var(--accent,var(--ink)); outline-offset:2px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>

<div class="wrap">
  <header class="top">
    <p class="eyebrow">Teacher distillation · MeetingBank</p>
    <h1>Four teachers, one transcript</h1>
    <p class="lede">Each model was asked to compress the same council-meeting transcripts by
    deleting words, never rewriting them. Below, every passage is shown <em>as the aligner sees
    it</em> — the source with each model's discards struck through. What a model throws away is
    the training signal; what it invents is the noise.</p>
  </header>

  <div class="tablewrap">
    <table>
      <caption>Scored over 12 passages × 3 samples. Invented words counted across all 12,
      after stripping any title the model added.</caption>
      <thead>
        <tr><th scope="col" style="text-align:left">Teacher</th><th scope="col">ratio</th>
        <th scope="col">variation</th><th scope="col">align gap</th><th scope="col">negation</th>
        <th scope="col">numbers</th><th scope="col">order</th><th scope="col">invented</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>

  <div class="legend">
    <span><b class="sample-d">struck</b> discarded</span>
    <span><b>solid</b> kept</span>
    <span><span class="flag">not in source</span> a word the model introduced</span>
    <label class="toggle"><input type="checkbox" id="dropToggle" checked> Show discarded words</label>
  </div>

  {"".join(panels)}

  <section class="finding">
    <h2>The closing line of Passage 9</h2>
    <p>The transcript ends <code>Ocean carries.</code> — a speech-recognition error for
    <em>Motion carries</em>. Gemini, Haiku and Kimi all kept the mistake. Sonnet quietly
    corrected it.</p>
    <p>That instinct is right for a summariser and wrong here. These outputs become training
    labels by aligning each output word back to a source position, so a word that never
    appeared in the source has no position to map to: it is dropped, or it drags the alignment
    out of step. Across the 12 passages Sonnet introduced <b>57</b> such words against
    Gemini's 12, Haiku's 14 and Kimi's 5 — and Sonnet is the model that most improves the text
    it is copying.</p>
  </section>

  <footer>Outputs sampled at temperature 0.7; the first sample of each passage is shown.
  Markup produced by the same alignment code the training pipeline uses.</footer>
</div>

<script>
  const body = document.body, box = document.getElementById('dropToggle');
  const sync = () => body.classList.toggle('show-drops', box.checked);
  box.addEventListener('change', sync);
  sync();
</script>
"""

dest = pathlib.Path(
    "/Users/akshayballal/Developer/Projects/Starlight/grug"
    "/.claude/worktrees/negation-metric-fix/benchmarks/teacher-examples.html"
)
dest.write_text(page)
print(f"wrote {dest}  ({len(page) / 1024:.0f} KB)")
