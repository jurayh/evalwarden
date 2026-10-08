"""Trace observatory reporter: one self-contained HTML page per scan.

The page is the observatory's whole user interface: a phase timeline
per session, the priced waste table, and cross-session rollups. Like
the other reporters it is a single file with inline CSS, no JavaScript
and no remote assets, so it can be committed or attached anywhere.

All numbers come from the deterministic analyses
(:mod:`evalwarden.trace_phases`, :mod:`evalwarden.trace_waste`); the
renderer computes nothing of its own beyond presentation sums. Prices
are estimates at the configured per-1M rates and are labeled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from jinja2 import Environment

from ..model import IntegrityModel
from ..trace_phases import PHASES, PhaseSegment, segment_phases
from ..trace_waste import WasteReport, analyze_waste

PHASE_COLORS = {
    "explore": "#4c78a8",
    "plan": "#e6c229",
    "edit": "#54a24b",
    "test": "#e45756",
    "review": "#72b7b2",
    "other": "#bab0ac",
}
PHASE_TEXT_DARK = {"plan", "other"}  # light fills need dark text


@dataclass
class SessionView:
    agent: str
    source_label: str
    session_id: str
    prompt: str
    started_at: str | None
    duration_s: float | None
    model_id: str | None
    tokens_in: int | None
    tokens_out: int | None
    spend_usd: float | None  # None when the session recorded no usage
    segments: list[PhaseSegment] = field(default_factory=list)
    waste: WasteReport | None = None

    @property
    def steps(self) -> int:
        return sum(s.length for s in self.segments)


@dataclass
class TraceView:
    sessions: list[SessionView]
    generated_at: str
    total_tokens_in: int
    total_tokens_out: int
    total_spend_usd: float
    unpriced_sessions: int  # sessions whose spend could not be computed
    waste: WasteReport  # merged across sessions
    phase_mix: list[tuple[str, int, float]]  # (label, steps, share) in PHASES order
    worst_loops: list  # LoopWaste entries, priced first then by wasted calls
    notes: list[str]  # deduplicated coverage notes across models


def _spend(tokens_in: int | None, tokens_out: int | None,
           price_in: float, price_out: float) -> float | None:
    if tokens_in is None or tokens_out is None:
        return None
    return round(tokens_in / 1_000_000 * price_in + tokens_out / 1_000_000 * price_out, 6)


def build_trace_view(
    models: list[IntegrityModel],
    source_labels: dict[str, str] | None = None,
    price_in_per_1m: float = 3.0,
    price_out_per_1m: float = 15.0,
) -> TraceView:
    """Assemble the page model from imported trace models."""
    source_labels = source_labels or {}
    sessions: list[SessionView] = []
    merged_waste = WasteReport(price_in_per_1m=price_in_per_1m,
                               price_out_per_1m=price_out_per_1m)
    phase_steps: dict[str, int] = {label: 0 for label in PHASES}
    notes: list[str] = []
    for model in models:
        label = source_labels.get(model.eval_id, model.eval_id)
        prompts = {t.id: t.prompt for t in model.tasks}
        starts = {t.id: t.metadata.get("started_at") for t in model.tasks}
        model_waste = analyze_waste(model, price_in_per_1m, price_out_per_1m)
        waste_by_task: dict[str, WasteReport] = {}
        for loop in model_waste.loops:
            waste_by_task.setdefault(loop.task_id, WasteReport()).loops.append(loop)
        for unused in model_waste.unused:
            waste_by_task.setdefault(unused.task_id, WasteReport()).unused.append(unused)
        for attempt in model.attempts:
            segments = segment_phases(attempt.spans)
            for segment in segments:
                phase_steps[segment.label] += segment.length
            sessions.append(SessionView(
                agent=model.adapter_name,
                source_label=label,
                session_id=attempt.task_id,
                prompt=prompts.get(attempt.task_id, ""),
                started_at=starts.get(attempt.task_id),
                duration_s=attempt.latency_s,
                model_id=attempt.model_id,
                tokens_in=attempt.tokens_in,
                tokens_out=attempt.tokens_out,
                spend_usd=_spend(attempt.tokens_in, attempt.tokens_out,
                                 price_in_per_1m, price_out_per_1m),
                segments=segments,
                waste=waste_by_task.get(attempt.task_id),
            ))
        merged_waste.loops.extend(model_waste.loops)
        merged_waste.unused.extend(model_waste.unused)
        merged_waste.total_wasted_tokens_in += model_waste.total_wasted_tokens_in
        merged_waste.total_wasted_tokens_out += model_waste.total_wasted_tokens_out
        merged_waste.total_wasted_usd += model_waste.total_wasted_usd
        merged_waste.unpriced_wasted_calls += model_waste.unpriced_wasted_calls
        for note in model.unsupported:
            if note not in notes:
                notes.append(note)
    merged_waste.total_wasted_usd = round(merged_waste.total_wasted_usd, 6)
    total_steps = sum(phase_steps.values())
    phase_mix = [
        (label, phase_steps[label], (phase_steps[label] / total_steps if total_steps else 0.0))
        for label in PHASES if phase_steps[label] > 0
    ]
    worst = sorted(
        merged_waste.loops,
        key=lambda w: (w.priced, w.usd or 0.0, w.wasted_calls),
        reverse=True,
    )[:10]
    sessions.sort(key=lambda s: (s.started_at or "", s.session_id))
    return TraceView(
        sessions=sessions,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_tokens_in=sum(s.tokens_in or 0 for s in sessions),
        total_tokens_out=sum(s.tokens_out or 0 for s in sessions),
        total_spend_usd=round(sum(s.spend_usd or 0.0 for s in sessions), 6),
        unpriced_sessions=sum(1 for s in sessions if s.spend_usd is None),
        waste=merged_waste,
        phase_mix=phase_mix,
        worst_loops=worst,
        notes=notes,
    )


def format_usd(value: float) -> str:
    """Dollar format that keeps sub-cent waste figures exact.

    Trace waste lives below a cent, so small amounts keep up to six
    decimals (trailing zeros stripped, two always shown); larger
    amounts round to cents.
    """
    if value >= 10:
        return f"${value:,.2f}"
    decimals = 6 if value < 0.1 else 4
    text = f"{value:,.{decimals}f}".rstrip("0")
    if text.endswith("."):
        text += "00"
    elif "." in text and len(text.split(".")[1]) == 1:
        text += "0"
    return f"${text}"


def _usd(value: float | None) -> str:
    return "unpriced" if value is None else format_usd(value)


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace observatory</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; color: #1a1a1a; background: #f4f4f4; line-height: 1.5; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 32px 24px 64px; }
  .card { background: #fff; border: 1px solid #e3e3e3; border-radius: 12px;
          padding: 22px 24px; margin-bottom: 16px; }
  .masthead h1 { margin: 4px 0 2px; font-size: 26px; }
  .kicker { font-size: 12px; font-weight: 700; letter-spacing: .12em;
            text-transform: uppercase; color: #666; }
  .meta { color: #666; font-size: 13px; }
  .stats { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; }
  .stat { flex: 1 1 150px; background: #fafafa; border: 1px solid #eee; border-radius: 10px;
          padding: 12px 14px; }
  .stat .num { font-size: 22px; font-weight: 800; }
  .stat .lbl { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .06em; }
  .bar { display: flex; height: 26px; border-radius: 6px; overflow: hidden; margin: 10px 0 4px; }
  .bar > div { display: flex; align-items: center; justify-content: center;
               font-size: 11px; font-weight: 700; color: #fff; min-width: 0;
               white-space: nowrap; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  table th, table td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #f0f0f0; }
  table th { color: #666; font-weight: 600; }
  code { background: #f4f4f4; padding: 1px 6px; border-radius: 4px; font-size: 12.5px;
         word-break: break-all; }
  .prompt { font-size: 15px; margin: 2px 0 6px; }
  .waste-num { color: #b3261e; font-weight: 700; }
  ul.notes { margin: 6px 0; padding-left: 20px; font-size: 13px; color: #444; }
  .legend { font-size: 12px; color: #555; }
  .legend span { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
                 margin: 0 4px 0 12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card masthead">
    <div class="kicker">EvalWarden · Trace observatory</div>
    <h1>Where your coding agents spent, and how the work flowed</h1>
    <p class="meta">Generated {{ view.generated_at }} · read locally from session files ·
      nothing was uploaded · dollar figures are estimates at
      ${{ "%.2f" % view.waste.price_in_per_1m }} / ${{ "%.2f" % view.waste.price_out_per_1m }}
      per 1M tokens in/out, exact wherever the format records per-step usage.</p>
    <div class="stats">
      <div class="stat"><div class="num">{{ view.sessions | length }}</div><div class="lbl">sessions</div></div>
      <div class="stat"><div class="num">{{ "{:,}".format(view.total_tokens_in + view.total_tokens_out) }}</div><div class="lbl">tokens in + out</div></div>
      <div class="stat"><div class="num">{{ usd(view.total_spend_usd) }}</div><div class="lbl">estimated spend</div></div>
      <div class="stat"><div class="num waste-num">{{ usd(view.waste.total_wasted_usd) }}</div><div class="lbl">wasted on loops</div></div>
      <div class="stat"><div class="num">{{ view.waste.total_wasted_calls }}</div><div class="lbl">wasted calls</div></div>
    </div>
    {% if view.unpriced_sessions %}
    <p class="meta">{{ view.unpriced_sessions }} session(s) recorded no token usage; their spend is not estimated.</p>
    {% endif %}
    {% if view.waste.unpriced_wasted_calls %}
    <p class="meta">{{ view.waste.unpriced_wasted_calls }} wasted call(s) have no recorded per-step
      usage and are reported unpriced rather than estimated.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Phase mix (all sessions, by tool call)</h2>
    {% if view.phase_mix %}
    <div class="bar">
      {% for label, steps, share in view.phase_mix %}
      <div style="width: {{ "%.2f" % (share * 100) }}%; background: {{ colors[label] }}{% if label in dark_text %}; color: #333{% endif %}"
           title="{{ label }}: {{ steps }} calls">{{ label if share >= 0.09 else "" }}</div>
      {% endfor %}
    </div>
    <p class="legend">{% for label, steps, share in view.phase_mix %}<span style="background: {{ colors[label] }}"></span>{{ label }} {{ steps }} ({{ "%.0f" % (share * 100) }}%){% endfor %}</p>
    {% else %}
    <p class="meta">No tool calls recorded.</p>
    {% endif %}
  </div>

  {% if view.worst_loops %}
  <div class="card">
    <h2>Worst loops</h2>
    <table>
      <tr><th>Repeated call</th><th>Session</th><th>Times</th><th>Wasted calls</th><th>Wasted tokens</th><th>Wasted $</th></tr>
      {% for w in view.worst_loops %}
      <tr>
        <td><code>{{ w.key | truncate(72) }}</code></td>
        <td><code>{{ w.task_id | truncate(24) }}</code></td>
        <td>{{ w.total }}x</td><td>{{ w.wasted_calls }}</td>
        <td>{{ "{:,}".format(w.tokens_in + w.tokens_out) if w.priced else "—" }}</td>
        <td class="waste-num">{{ usd(w.usd) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  {% for s in view.sessions %}
  <div class="card">
    <div class="kicker">{{ s.agent }} · {{ s.source_label }}</div>
    <p class="prompt">{{ s.prompt | truncate(180) or "(no prompt recorded)" }}</p>
    <p class="meta">
      session <code>{{ s.session_id | truncate(40) }}</code>
      {% if s.started_at %} · started {{ s.started_at[:16].replace("T", " ") }}{% endif %}
      {% if s.duration_s is not none %} · {{ "%.0f" % s.duration_s }}s{% endif %}
      {% if s.model_id %} · {{ s.model_id }}{% endif %}
      · {{ s.steps }} tool calls
      {% if s.tokens_in is not none %} · {{ "{:,}".format(s.tokens_in) }} in / {{ "{:,}".format(s.tokens_out) }} out tokens{% endif %}
      {% if s.spend_usd is not none %} · est. {{ usd(s.spend_usd) }}{% endif %}
    </p>
    {% if s.segments %}
    <div class="bar">
      {% for seg in s.segments %}
      <div style="width: {{ "%.2f" % (seg.length / s.steps * 100) }}%; background: {{ colors[seg.label] }}{% if seg.label in dark_text %}; color: #333{% endif %}"
           title="{{ seg.label }}: steps {{ seg.start + 1 }}-{{ seg.end }}">{{ seg.label if seg.length / s.steps >= 0.12 else "" }}</div>
      {% endfor %}
    </div>
    <table>
      <tr><th>Phase</th><th>Steps</th><th>Calls</th><th>Tools</th></tr>
      {% for seg in s.segments %}
      <tr><td>{{ seg.label }}</td><td>{{ seg.start + 1 }}–{{ seg.end }}</td><td>{{ seg.length }}</td>
          <td>{{ seg.tools | unique | join(", ") }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
    {% if s.waste and (s.waste.loops or s.waste.unused) %}
    <table>
      <tr><th>Waste</th><th>Detail</th><th>Wasted $</th></tr>
      {% for w in s.waste.loops %}
      <tr><td>loop</td><td><code>{{ w.key | truncate(64) }}</code> ×{{ w.total }} ({{ w.wasted_calls }} wasted)</td>
          <td class="waste-num">{{ usd(w.usd) }}</td></tr>
      {% endfor %}
      {% for u in s.waste.unused %}
      <tr><td>unused outputs</td><td>{{ u.step_ids | length }} outputs consumed by nothing downstream</td>
          <td class="waste-num">{{ usd(u.usd) }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
  </div>
  {% endfor %}

  <div class="card">
    <h2>What this page cannot see</h2>
    <p class="meta">Silence here means "the format does not record it", never "it did not happen".
      The imports reported these coverage limits:</p>
    <ul class="notes">
      {% for note in view.notes[:14] %}
      <li>{{ note }}</li>
      {% endfor %}
    </ul>
    {% if view.notes | length > 14 %}
    <p class="meta">…and {{ view.notes | length - 14 }} more coverage note(s).</p>
    {% endif %}
  </div>
</div>
</body>
</html>
"""


def render_trace_report(view: TraceView, version: str | None = None) -> str:
    env = Environment(autoescape=True)
    env.filters["unique"] = lambda items: list(dict.fromkeys(items))
    template = env.from_string(_TEMPLATE)
    return template.render(
        view=view, colors=PHASE_COLORS, dark_text=PHASE_TEXT_DARK, usd=_usd,
        version=version,
    )
