"""Render the open Khan Academy exercise templates into concrete problems with worked solutions.

Each template in https://github.com/Khan/khan-exercises is an HTML file that generates random problem instances in
the browser. This tool loads every template in headless Edge/Chrome, lets the framework build one instance per
(problem type, seed), reveals all step-by-step hints and writes one JSON line per instance.

Why this corpus: Junyi Academy (2015 log) was built on the same templates, so a Junyi exercise name equals a template
name. That gives the item-to-document key ASSISTments lacks. Exercise content is CC BY-NC-SA 3.0 (non-commercial).

Setup (once):
    git clone https://github.com/Khan/khan-exercises.git <dir>
    git -C <dir> checkout 001b27a5^      # last commit that still ships the local-mode runner (May 2016)
    python -m pip install playwright     # uses the installed Edge; no browser download needed

Usage:
    python -m lmsrag.data_tools.render_khan --repo <dir> --out lmsrag/data/khan_v2/khan_rendered.jsonl --seeds 4
Rendering is deterministic: the framework seeds its random generator from the ``seed`` query parameter.
Re-running resumes: templates already present in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Runs before any page script: capture the solution element of the "newProblem" event the framework fires.
INIT_JS = r"""
(function(){
  var t = setInterval(function(){
    if (window.jQuery && window.Exercises && !window.__bound) {
      window.__bound = true;
      jQuery(window.Exercises).bind('newProblem', function(e, d){
        var el = null, typ = null;
        try { el = d.solution && d.solution[0] ? d.solution[0].cloneNode(true) : null; } catch(err) {}
        try { typ = d.answerType || null; } catch(err) {}
        window.__sol = el; window.__np = {n: (window.__np ? window.__np.n : 0) + 1, answer_type: typ};
      });
      clearInterval(t);
    }
  }, 1);
})();
"""

# Text extraction keeps the TeX source of every formula as $...$ and drops drawings and widgets.
EXTRACT_JS = r"""() => {
  function txt(el){
    if(!el) return '';
    var c = el.cloneNode(true);
    c.querySelectorAll('[data-math-formula]').forEach(function(n){ n.replaceWith(document.createTextNode(' $' + n.getAttribute('data-math-formula') + '$ ')); });
    c.querySelectorAll('code').forEach(function(n){ n.replaceWith(document.createTextNode(' $' + n.textContent + '$ ')); });
    c.querySelectorAll('.graphie, svg, script, style, .vars, .MathJax, .katex, input, button, .hints').forEach(function(n){ n.remove(); });
    c.querySelectorAll('p, li, div, br, tr').forEach(function(n){ n.appendChild(document.createTextNode('\n')); });
    return c.textContent.replace(/[ \t ]+/g,' ').replace(/ ?\n[ \n]*/g,'\n').trim();
  }
  var wa = document.querySelector('#workarea');
  var q = wa ? Array.from(wa.querySelectorAll('.problem, .question')).map(txt).filter(function(s){return s.length>0;}).join('\n') : '';
  var hints = Array.from(document.querySelectorAll('#hintsarea > *')).map(txt).filter(function(s){return s.length>0;});
  var choices = Array.from(document.querySelectorAll('#solutionarea li')).map(txt).filter(function(s){return s.length>0;});
  var seed = null; try { seed = Khan.getSeedInfo(); } catch(e) {}
  var sa = document.querySelector('#solutionarea'), w = '';
  if (sa) { w = sa.querySelector('input[type=radio]') ? 'radio' : sa.querySelector('input[type=checkbox]') ? 'checkbox' : sa.querySelector('select') ? 'select' : sa.querySelector('input') ? 'input' : 'other'; }
  return { question: q || txt(wa), workarea_text: txt(wa),
           has_figure: !!(wa && wa.querySelector('.graphie, svg, img')),
           hints: hints, choices: choices, answer_widget: w, answer_type: (window.__np||{}).answer_type,
           answer: txt(window.__sol), problem_type: seed ? seed.problem_type : null, title: document.title };
}"""

CLICK_HINT_JS = ("() => { var h = document.querySelector('#hint'); "
                 "if (!h || h.disabled || h.classList.contains('disabled')) return false; jQuery(h).click(); return true; }")


def problem_type_count(repo: Path, slug: str, cap: int = 6) -> int:
    """Number of problem types a template defines (each has its own solution block), capped."""
    html = (repo / "exercises" / f"{slug}.html").read_text(encoding="utf-8", errors="ignore")
    return max(1, min(cap, len(re.findall(r'class="solution"', html))))


def render_one(ctx, port: int, slug: str, problem_index: int, seed: int) -> dict:
    record = {"slug": slug, "problem_index": problem_index, "seed": seed}
    for attempt in range(3):  # the first load of a page occasionally loses a script-loader race
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)[:160]))
        try:
            page.goto(f"http://127.0.0.1:{port}/exercises/{slug}.html?seed={seed}&problem={problem_index}",
                      wait_until="load", timeout=30000)
            page.wait_for_function("window.__np && window.__np.n >= 1", timeout=12000)
            page.wait_for_timeout(200)
            for _ in range(40):
                if not page.evaluate(CLICK_HINT_JS):
                    break
                page.wait_for_timeout(30)
            page.wait_for_timeout(200)
            record.update(page.evaluate(EXTRACT_JS))
            record.pop("fail", None)
            record["attempts"] = attempt + 1
            page.close()
            return record
        except Exception as exc:  # recorded per item; rendering continues
            record["fail"] = (errors[0] if errors else str(exc))[:160]
            try:
                page.close()
            except Exception:
                pass
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path, help="khan-exercises checkout at commit 001b27a5^")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=4, help="instances per problem type")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--slugs", type=Path, help="optional text file with one template name per line")
    parser.add_argument("--channel", default="msedge", help="installed browser channel: msedge or chrome")
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright  # optional dependency, imported lazily

    if args.slugs:
        slugs = [line.strip() for line in args.slugs.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        slugs = sorted(p.stem for p in (args.repo / "exercises").glob("*.html") if not p.stem.startswith("khan-"))
    done = set()
    if args.out.exists():
        done = {json.loads(line)["slug"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen([sys.executable, "-m", "http.server", str(args.port), "--bind", "127.0.0.1"],
                              cwd=args.repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    ok = failed = 0
    started = time.time()
    try:
        with sync_playwright() as p, args.out.open("a", encoding="utf-8") as sink:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            ctx = browser.new_context()
            ctx.add_init_script(INIT_JS)
            for i, slug in enumerate(slugs, 1):
                if slug in done:
                    continue
                for problem_index in range(problem_type_count(args.repo, slug)):
                    for k in range(1, args.seeds + 1):
                        record = render_one(ctx, args.port, slug, problem_index, 37 * k + problem_index)
                        ok += "fail" not in record
                        failed += "fail" in record
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                print(f"[{i}/{len(slugs)}] {slug}: ok={ok} fail={failed} t={time.time() - started:.0f}s", flush=True)
            browser.close()
    finally:
        server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
