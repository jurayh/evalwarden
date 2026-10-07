"""Report card reporter: a shareable one-page integrity summary per benchmark.

A report card is the public face of an audit: overall verdict, per-category
scores, key findings, and a methodology footer -- in one self-contained HTML
file (inline CSS, no JavaScript, no remote assets) that can be committed next
to benchmark results or attached to a release note.

Unlike the full HTML report, the card is a summary, not the evidence dump:
it shows the top findings per category and points at the full report for the
rest. Category scores reuse the engine's fixed severity deductions, so a
card and a full report can never disagree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from jinja2 import Environment

from ..checks import BY_ID
from ..engine import AuditResult, integrity_score
from ..model import SEVERITY_ORDER, Severity

# Check-ID prefix -> display category. A new check family is one line here.
CATEGORIES: dict[str, str] = {
    "ENV": "Environment",
    "GRAD": "Grader",
    "JUDGE": "Judge",
    "COST": "Cost and efficiency",
}


def category_of(check_id: str) -> str:
    """Map a check ID like "ENV-001" to its display category."""
    return CATEGORIES.get(check_id.split("-")[0], "Other")


# Lane order and labels for the "Checks run" listing: the reader scans by
# what part of the eval is being checked, so groups follow audit order.
LANES: list[tuple[str, str]] = [
    ("ENV", "Environment"),
    ("GRAD", "Grader"),
    ("COST", "Cost and efficiency"),
    ("JUDGE", "Judge"),
    ("DATA", "Dataset"),
    ("TRAJ", "Trajectory"),
    ("NOISE", "Noise budget"),
]


def grouped_checks() -> list[tuple[str, list[tuple[str, str]]]]:
    """All registered checks as (lane label, [(id, title), ...]) in lane order."""
    by_prefix: dict[str, list[tuple[str, str]]] = {}
    for cid in sorted(BY_ID):
        by_prefix.setdefault(cid.split("-")[0], []).append((cid, BY_ID[cid].meta.title))
    groups = [(label, by_prefix[prefix]) for prefix, label in LANES if prefix in by_prefix]
    groups += [
        (prefix, items) for prefix, items in by_prefix.items() if prefix not in dict(LANES)
    ]
    return groups


@dataclass
class CategoryScore:
    name: str
    score: int  # 0-100, same deduction table as the overall score
    errors: int
    highs: int
    mediums: int
    lows: int


@dataclass
class CardEntry:
    eval_id: str
    verdict: str  # "BLOCKED" | "PASS"
    score: int
    filename: str
    generated_at: str


def category_scores(result: AuditResult) -> list[CategoryScore]:
    """Per-category scores, in CATEGORIES order then any leftovers."""
    by_category: dict[str, list] = {}
    for finding in result.findings:
        by_category.setdefault(category_of(finding.id), []).append(finding)
    scores: list[CategoryScore] = []
    ordered = list(CATEGORIES.values()) + [
        name for name in by_category if name not in CATEGORIES.values()
    ]
    for name in ordered:
        findings = by_category.get(name, [])
        counts = {sev: 0 for sev in Severity}
        for f in findings:
            counts[f.severity] += 1
        scores.append(
            CategoryScore(
                name=name,
                score=integrity_score(findings),
                errors=counts[Severity.ERROR],
                highs=counts[Severity.HIGH],
                mediums=counts[Severity.MEDIUM],
                lows=counts[Severity.LOW],
            )
        )
    return scores


def slugify(eval_id: str) -> str:
    """Filesystem-safe card filename stem for an eval id."""
    return re.sub(r"[^a-z0-9]+", "-", eval_id.lower()).strip("-") or "eval"


_CARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Integrity report card: {{ model.eval_id }}</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; color: #1a1a1a; background: #f4f4f4; line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 32px 24px 64px; }
  .card { background: #fff; border: 1px solid #e3e3e3; border-radius: 12px;
          padding: 24px 26px; margin-bottom: 16px; }
  .masthead { text-align: center; padding: 28px 26px; }
  .masthead .kicker { font-size: 12px; font-weight: 700; letter-spacing: .12em;
                      text-transform: uppercase; color: #666; }
  .masthead h1 { margin: 6px 0 2px; font-size: 24px; }
  .score { font-size: 56px; font-weight: 800; line-height: 1.1; }
  .verdict { display: inline-block; font-weight: 800; font-size: 14px; letter-spacing: .08em;
             border-radius: 8px; padding: 4px 14px; margin-top: 8px; }
  .verdict.blocked { background: #b3261e; color: #fff; }
  .verdict.pass { background: #1e7b34; color: #fff; }
  table.cats { width: 100%; border-collapse: collapse; font-size: 14px; }
  table.cats td { padding: 10px 4px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
  table.cats tr:last-child td { border-bottom: none; }
  .catname { width: 32%; font-weight: 600; }
  .catscore { width: 12%; text-align: right; font-variant-numeric: tabular-nums; }
  .bar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; min-width: 120px; }
  .bar i { display: block; height: 100%; border-radius: 4px; }
  .bar i.good { background: #1e7b34; } .bar i.warn { background: #d99a00; } .bar i.bad { background: #b3261e; }
  .counts { color: #888; font-size: 12px; white-space: nowrap; }
  h2 { margin: 0 0 10px; font-size: 16px; }
  .finding { border-top: 1px solid #f0f0f0; padding: 12px 0; }
  .finding:first-of-type { border-top: none; }
  .badge { display: inline-block; font-size: 11px; font-weight: 700; border-radius: 6px;
           padding: 2px 8px; margin-right: 6px; text-transform: uppercase; letter-spacing: .04em; }
  .sev-error { background: #fdecea; color: #b3261e; border: 1px solid #f5c6c2; }
  .sev-high { background: #fff4e5; color: #8a5a00; border: 1px solid #f0d9a8; }
  .sev-medium { background: #eef4ff; color: #1a56db; border: 1px solid #c9dbff; }
  .sev-low { background: #f1f1f1; color: #555; border: 1px solid #ddd; }
  .conf { background: #f1f1f1; color: #444; border: 1px solid #ddd; text-transform: none; }
  .finding .title { font-weight: 600; }
  .finding .ev { color: #555; font-size: 13.5px; margin: 4px 0 0; }
  .code { font-family: ui-monospace, monospace; font-size: 11px; color: #777;
          background: #f6f6f6; border: 1px solid #e6e6e6; border-radius: 5px;
          padding: 0 5px; white-space: nowrap; }
  .checkgroup { margin: 0 0 9px; }
  .checkgroup:last-child { margin-bottom: 0; }
  .lanelabel { font-size: 11px; font-weight: 700; letter-spacing: .08em;
               text-transform: uppercase; color: #888; }
  ul.checks { list-style: none; margin: 3px 0 0; padding: 0; }
  ul.checks li { padding: 1px 0; }
  .meta { color: #666; font-size: 13px; }
  .method td, .method th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f0f0;
                           font-size: 13px; vertical-align: top; }
  .method th { color: #666; font-weight: 600; white-space: nowrap; }
  .fp { font-family: ui-monospace, monospace; font-size: 12px; color: #888; }
  footer { margin-top: 24px; color: #888; font-size: 12.5px; text-align: center; }
</style>
</head>
<body>
<div class="wrap">

  <div class="card masthead">
    <div class="kicker">Integrity report card</div>
    <h1>{{ model.eval_id }}</h1>
    <div class="score">{{ result.score }}<span style="font-size:22px;color:#888">/100</span></div>
    <div><span class="verdict {{ 'blocked' if result.verdict == 'BLOCKED' else 'pass' }}">{{ result.verdict }}</span></div>
    <p class="meta" style="margin-bottom:0">
      {% if result.verdict == "BLOCKED" %}
      Blocked by {{ blocked_labels|join(", ") }}: the reported score cannot be trusted until these are fixed.
      {% else %}
      No blocking findings observed under this policy.
      {% endif %}
    </p>
  </div>

  <div class="card">
    <h2>Category scores</h2>
    <table class="cats">
      {% for c in categories %}
      <tr>
        <td class="catname">{{ c.name }}</td>
        <td><div class="bar"><i class="{{ 'good' if c.score >= 90 else 'warn' if c.score >= 60 else 'bad' }}" style="width:{{ c.score }}%"></i></div></td>
        <td class="catscore">{{ c.score }}</td>
        <td class="counts">{% if c.errors %}{{ c.errors }}E {% endif %}{% if c.highs %}{{ c.highs }}H {% endif %}{% if c.mediums %}{{ c.mediums }}M {% endif %}{% if c.lows %}{{ c.lows }}L{% endif %}{% if not (c.errors or c.highs or c.mediums or c.lows) %}clean{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="meta">Same fixed severity deductions as the full report: error -25, high -10, medium -5, low -1. Diagnostic, not a certification.</p>
  </div>

  <div class="card">
    <h2>Key findings ({{ key_findings|length }} of {{ result.findings|length }})</h2>
    {% if key_findings %}
    {% for f in key_findings %}
    <div class="finding">
      <span class="badge sev-{{ f.severity.value }}">{{ f.severity.value }}</span><span class="badge conf">{{ f.confidence.value }} confidence</span>
      <span class="title">{{ f.title }}</span> <span class="code">{{ f.id }}</span>
      {% if f.evidence %}<p class="ev">{{ f.evidence[0] }}</p>{% endif %}
    </div>
    {% endfor %}
    {% if result.findings|length > key_findings|length %}
    <p class="meta">Plus {{ result.findings|length - key_findings|length }} more in the full audit report.</p>
    {% endif %}
    {% else %}
    <p class="meta">None. This audit's checks found nothing to flag under this policy.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Methodology</h2>
    <table class="method">
      <tr><th>Tool</th><td>evalwarden {{ version }}: a linter for agent evaluations, not another eval framework. Read-only, offline; input files are never modified.</td></tr>
      <tr><th>Adapter</th><td>{{ model.adapter_name }} {{ model.adapter_version }}</td></tr>
      <tr><th>Checks run</th><td>{% for lane, items in check_groups %}<div class="checkgroup"><div class="lanelabel">{{ lane }}</div><ul class="checks">{% for id, title in items %}<li><span class="ct">{{ title }}</span> <span class="code">{{ id }}</span></li>{% endfor %}</ul></div>{% endfor %}</td></tr>
      <tr><th>Evidence</th><td class="fp">{% for fname, digest in model.digests.items() %}{{ fname }} sha256:{{ digest[:12] }}&hellip;{% if not loop.last %} {% endif %}{% endfor %}</td></tr>
      <tr><th>Generated</th><td>{{ generated_at }}</td></tr>
    </table>
    <p class="meta">Every finding carries a confidence label; precision over recall. Secret values are never stored in reports. Integrity scores are diagnostic, not a certification.</p>
  </div>

  <footer>Generated by evalwarden {{ version }} &middot; {{ generated_at }}</footer>

</div>
</body>
</html>
"""

KEY_FINDINGS_LIMIT = 5


def render_report_card(result: AuditResult, version: str | None = None) -> str:
    """Render a shareable one-page integrity report card as self-contained HTML."""
    if version is None:
        from .. import __version__ as pkg_version

        version = pkg_version
    env = Environment(autoescape=True)
    template = env.from_string(_CARD_TEMPLATE)
    blocked_titles: dict[str, str] = {}
    for f in result.blocked_by:
        blocked_titles.setdefault(f.id, f.title)
    return template.render(
        result=result,
        model=result.model,
        categories=category_scores(result),
        key_findings=result.findings[:KEY_FINDINGS_LIMIT],
        blocked_labels=[
            f"{blocked_titles[cid]} ({cid})" for cid in sorted(blocked_titles)
        ],
        check_groups=grouped_checks(),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        version=version,
    )


_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evalwarden integrity report cards</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; color: #1a1a1a; background: #f4f4f4; line-height: 1.5; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 22px; }
  .meta { color: #666; font-size: 13px; }
  table { width: 100%; border-collapse: collapse; background: #fff;
          border: 1px solid #e3e3e3; border-radius: 12px; overflow: hidden; font-size: 14px; }
  th, td { text-align: left; padding: 12px 16px; border-bottom: 1px solid #f0f0f0; }
  th { color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
  tr:last-child td { border-bottom: none; }
  a { color: #1a56db; text-decoration: none; } a:hover { text-decoration: underline; }
  .badge { display: inline-block; font-size: 11px; font-weight: 700; border-radius: 6px;
           padding: 2px 10px; text-transform: uppercase; letter-spacing: .06em; }
  .blocked { background: #fdecea; color: #b3261e; border: 1px solid #f5c6c2; }
  .pass { background: #e7f6ea; color: #1e7b34; border: 1px solid #bfe6c7; }
  .score { font-variant-numeric: tabular-nums; font-weight: 700; }
  footer { margin-top: 24px; color: #888; font-size: 12.5px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Integrity report cards</h1>
  <p class="meta">One-page integrity summaries generated by evalwarden. Each card is
  self-contained: verdict, category scores, key findings, methodology.</p>
  <table>
    <tr><th>Benchmark</th><th>Score</th><th>Verdict</th><th>Generated</th></tr>
    {% for c in cards %}
    <tr>
      <td><a href="{{ c.filename }}">{{ c.eval_id }}</a></td>
      <td class="score">{{ c.score }}/100</td>
      <td><span class="badge {{ 'blocked' if c.verdict == 'BLOCKED' else 'pass' }}">{{ c.verdict }}</span></td>
      <td class="meta">{{ c.generated_at }}</td>
    </tr>
    {% endfor %}
  </table>
  <footer>Generated by evalwarden {{ version }} &middot; {{ generated_at }}</footer>
</div>
</body>
</html>
"""


def render_index(cards: list[CardEntry], version: str | None = None) -> str:
    """Render an index page listing report cards. Self-contained HTML."""
    if version is None:
        from .. import __version__ as pkg_version

        version = pkg_version
    env = Environment(autoescape=True)
    template = env.from_string(_INDEX_TEMPLATE)
    return template.render(
        cards=cards,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        version=version,
    )
