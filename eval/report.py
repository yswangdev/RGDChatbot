"""
Render the evaluation deliverable: a per-question table grouped by evaluation
method (reference retrieval / reference generation / judge retrieval / judge
generation / robustness / combined), with an aggregate row, as HTML + CSV + MD.

Each column is tagged with the method it belongs to via a grouped header row
(HTML) and a method-label header row (CSV).
"""

import csv
import html
import os
from typing import Dict, List, Tuple

# (method group label, [(record key path, column header), ...])
GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("", [("id", "ID"), ("type", "Type")]),
    ("Question & answers", [
        ("question", "Question"),
        ("reference_answer", "Reference A"),
        ("rag_answer", "RAG A"),
    ]),
    ("Reference-based retrieval", [
        ("retrieval.precision_at_k", "P@k"),
        ("retrieval.recall_at_k", "R@k"),
        ("retrieval.mrr", "MRR"),
        ("retrieval.ndcg_at_k", "NDCG"),
        ("retrieval.context_recall", "CtxRecall"),
    ]),
    ("Reference-based generation", [
        ("generation.rougeL", "ROUGE-L"),
        ("generation.meteor", "METEOR"),
        ("generation.bertscore", "BERTScore"),
    ]),
    ("LLM-judge retrieval", [
        ("judge_retrieval.chunk_relevance_avg", "ChunkRel(0-3)"),
        ("judge_retrieval.context_quality", "CtxQual(1-5)"),
    ]),
    ("LLM-judge generation", [
        ("judge_generation.faithfulness", "Faith(1-5)"),
        ("judge_generation.correctness", "Correct(1-5)"),
        ("judge_generation.answer_relevance", "Relev(1-5)"),
        ("judge_generation.completeness", "Complete(1-5)"),
        ("judge_generation.rationale", "Judge rationale"),
    ]),
    ("Robustness", [
        ("robustness.stability", "Stability"),
    ]),
    ("Combined (normalized families + overall)", [
        ("families.retrieval_ref", "RefRetr"),
        ("families.generation_ref", "RefGen"),
        ("families.judge_retrieval", "JudgeRetr"),
        ("families.judge_generation", "JudgeGen"),
        ("families.robustness", "Robust"),
        ("overall", "OVERALL"),
    ]),
]

# Flattened column list.
COLUMNS = [(key, header) for _, cols in GROUPS for key, header in cols]
TEXT_KEYS = {"id", "type", "question", "reference_answer", "rag_answer", "judge_generation.rationale"}
# Only these normalized [0,1] columns get the heatmap.
COLOR_KEYS = {
    "families.retrieval_ref", "families.generation_ref", "families.judge_retrieval",
    "families.judge_generation", "families.robustness", "overall",
}


def _get(record: Dict, path: str):
    node = record
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_csv(records: List[Dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Row 1: method group labels, repeated across each group's span.
        group_row = []
        for label, cols in GROUPS:
            group_row.extend([label] * len(cols))
        w.writerow(group_row)
        # Row 2: column headers.
        w.writerow([h for _, h in COLUMNS])
        # Data rows.
        for r in records:
            w.writerow([_fmt(_get(r, key)) for key, _ in COLUMNS])


def write_markdown(records: List[Dict], summary: Dict, path: str):
    lines = ["# RAG Evaluation Report", ""]
    lines.append(f"**System overall: {summary.get('overall', 0):.3f}** over {summary.get('n', 0)} questions\n")
    lines.append("## Family means (normalized 0-1)")
    for fam, val in summary.get("family_means", {}).items():
        lines.append(f"- {fam}: {val:.3f}")
    lines.append("\n## Per-question overall")
    lines.append("| ID | Type | Overall |")
    lines.append("|----|------|---------|")
    for r in records:
        lines.append(f"| {r['id']} | {r.get('type','')} | {r['overall']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _color(key, v):
    # Only the normalized family scores and the overall get a heatmap; the raw
    # per-metric columns (different scales) are left uncolored.
    if key not in COLOR_KEYS:
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    r = int(255 * (1 - v)); g = int(180 * v)
    return f' style="background:rgba({r},{g},80,0.18)"'


def write_html(records: List[Dict], summary: Dict, path: str):
    cards = "".join(
        f'<div class="stat-card"><div class="number">{v:.3f}</div>'
        f'<div class="label">{html.escape(k)}</div></div>'
        for k, v in {**{"OVERALL": summary.get("overall", 0)}, **summary.get("family_means", {})}.items()
    )

    # Two-row header: method groups (colspan) then column names.
    group_th = "".join(
        f'<th colspan="{len(cols)}" class="grp">{html.escape(label)}</th>'
        for label, cols in GROUPS
    )
    # Family/overall columns get a clickable sort arrow.
    col_th_parts = []
    for idx, (key, h) in enumerate(COLUMNS):
        if key in COLOR_KEYS:
            col_th_parts.append(
                f'<th class="colh sortable" onclick="sortTable({idx})" title="Click to sort">'
                f'{html.escape(h)}<span class="arrow">⇅</span></th>'
            )
        else:
            col_th_parts.append(f'<th class="colh">{html.escape(h)}</th>')
    col_th = "".join(col_th_parts)

    def cell(record, key):
        v = _get(record, key)
        cls = ' class="qa"' if key in TEXT_KEYS and key not in ("id", "type") else ""
        return f"<td{cls}{_color(key, v)}>{html.escape(_fmt(v))}</td>"

    rows = "".join("<tr>" + "".join(cell(r, key) for key, _ in COLUMNS) + "</tr>" for r in records)

    # Aggregate row: mean of numeric columns, blanks for text columns.
    agg_cells = ""
    for key, _ in COLUMNS:
        vals = [_get(r, key) for r in records]
        vals = [v for v in vals if isinstance(v, (int, float))]
        agg_cells += f"<td><b>{(sum(vals)/len(vals)):.3f}</b></td>" if vals else "<td></td>"
    agg_row = f'<tr class="agg">{agg_cells}</tr>'

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>RGD RAG Evaluation</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f5f5;color:#222;margin:0;padding:20px}}
h1{{color:#1a237e}} .stats{{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}}
.stat-card{{background:#fff;border-radius:8px;padding:12px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.12)}}
.stat-card .number{{font-size:1.5em;font-weight:700;color:#1a237e}} .stat-card .label{{font-size:.72em;color:#666}}
/* width:max-content lets the table grow past the container so it scrolls
   horizontally; min-width:100% keeps it filling a narrow screen. */
table{{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;background:#fff;font-size:.78em}}
th,td{{border:1px solid #e0e0e0;padding:6px 8px;text-align:center;vertical-align:top}}
/* Numeric/score cells stay on one line so columns are readable. */
td,th.colh{{white-space:nowrap}}
/* Two sticky header rows: group row pinned at top, column row just below it. */
thead th{{position:sticky;z-index:2;background:#1a237e;color:#fff}}
th.grp{{top:0;height:30px;box-sizing:border-box;z-index:3;background:#0d1442;border-bottom:2px solid #fff;font-size:.9em}}
th.colh{{top:30px;z-index:2}}
/* Only the text columns wrap (and break long URLs); they get a fixed width. */
td.qa{{text-align:left;white-space:normal;overflow-wrap:anywhere;word-break:break-word;
  min-width:240px;max-width:340px;font-size:.95em}}
tr.agg td{{background:#fff8e1;border-top:2px solid #f0c000}}
th.sortable{{cursor:pointer}} th.sortable:hover{{background:#283593}}
.arrow{{font-size:.85em;margin-left:3px;color:#9fa8da}}
/* .wrap is the scroll container so the sticky header pins on vertical scroll. */
.wrap{{max-height:82vh;overflow:auto}} .legend{{font-size:.8em;color:#666;margin:8px 0}}
</style></head><body>
<h1>RGD RAG Evaluation</h1>
<div class="stats">{cards}</div>
<div class="legend">Heatmap applies to normalized [0,1] scores. Judge columns are raw scales
(0-3 or 1-5). Aggregate row = mean of each numeric column over {len(records)} questions.
Click a ⇅ column header to sort.</div>
<div class="wrap"><table>
<thead><tr>{group_th}</tr><tr>{col_th}</tr></thead>
<tbody>{agg_row}{rows}</tbody>
</table></div>
<script>
let sortDir = {{}};
function sortTable(col) {{
  const tb = document.querySelector('table tbody');
  const rows = Array.from(tb.rows);
  const agg = rows.filter(r => r.classList.contains('agg'));   // keep pinned on top
  const data = rows.filter(r => !r.classList.contains('agg'));
  const dir = sortDir[col] === 'desc' ? 'asc' : 'desc';
  sortDir = {{}}; sortDir[col] = dir;
  const num = t => {{ const v = parseFloat(t); return isNaN(v) ? -Infinity : v; }};
  data.sort((a, b) => {{
    const x = num(a.cells[col].textContent), y = num(b.cells[col].textContent);
    return dir === 'asc' ? x - y : y - x;
  }});
  tb.replaceChildren(...agg, ...data);
  document.querySelectorAll('th.sortable .arrow').forEach(a => a.textContent = '⇅');
  const ths = document.querySelectorAll('thead tr:last-child th');
  const arrow = ths[col] && ths[col].querySelector('.arrow');
  if (arrow) arrow.textContent = dir === 'asc' ? '▲' : '▼';
}}
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def render_report(records: List[Dict], summary: Dict, out_dir: str):
    write_csv(records, os.path.join(out_dir, "eval_results.csv"))
    write_html(records, summary, os.path.join(out_dir, "eval_report.html"))
    write_markdown(records, summary, os.path.join(out_dir, "eval_report.md"))
