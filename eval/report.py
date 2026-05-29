"""
Render the evaluation deliverable: a per-question table (Q | reference A |
RAG A | each score | overall) plus an aggregate row, as HTML + CSV + Markdown.
"""

import csv
import html
import json
import os
from typing import Dict, List

# Flat metric columns shown in the table (key path -> short header).
SCORE_COLUMNS = [
    ("retrieval.precision_at_k", "P@k"),
    ("retrieval.recall_at_k", "R@k"),
    ("retrieval.hit_rate", "Hit"),
    ("retrieval.mrr", "MRR"),
    ("retrieval.ndcg_at_k", "NDCG"),
    ("retrieval.context_recall", "CtxRec"),
    ("generation.bleu", "BLEU"),
    ("generation.rouge1", "R-1"),
    ("generation.rouge2", "R-2"),
    ("generation.rougeL", "R-L"),
    ("generation.meteor", "METEOR"),
    ("generation.token_f1", "TokF1"),
    ("generation.bertscore", "BERTSc"),
    ("judge_retrieval.chunk_relevance_avg", "J:ChunkRel(0-3)"),
    ("judge_retrieval.context_quality", "J:CtxQual(1-5)"),
    ("judge_generation.faithfulness", "J:Faith(1-5)"),
    ("judge_generation.correctness", "J:Correct(1-5)"),
    ("judge_generation.answer_relevance", "J:Relev(1-5)"),
    ("judge_generation.completeness", "J:Complete(1-5)"),
    ("robustness.stability", "Robust"),
    ("families.retrieval_ref", "F:RetrRef"),
    ("families.generation_ref", "F:GenRef"),
    ("families.judge_retrieval", "F:JRetr"),
    ("families.judge_generation", "F:JGen"),
    ("families.robustness", "F:Robust"),
    ("overall", "OVERALL"),
]


def _get(record: Dict, path: str):
    node = record
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_csv(records: List[Dict], path: str):
    headers = ["id", "type", "question", "reference_answer", "rag_answer"] + [h for _, h in SCORE_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in records:
            row = [r["id"], r.get("type", ""), r["question"], r["reference_answer"], r["rag_answer"]]
            row += [_fmt(_get(r, key)) for key, _ in SCORE_COLUMNS]
            w.writerow(row)


def write_markdown(records: List[Dict], summary: Dict, path: str):
    lines = ["# RAG Evaluation Report", ""]
    lines.append(f"**System overall: {summary.get('overall', 0):.3f}** over {summary.get('n', 0)} questions\n")
    lines.append("## Family means")
    for fam, val in summary.get("family_means", {}).items():
        lines.append(f"- {fam}: {val:.3f}")
    lines.append("\n## Per-question overall")
    lines.append("| ID | Type | Overall |")
    lines.append("|----|------|---------|")
    for r in records:
        lines.append(f"| {r['id']} | {r.get('type','')} | {r['overall']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_html(records: List[Dict], summary: Dict, path: str):
    cards = "".join(
        f'<div class="stat-card"><div class="number">{v:.3f}</div>'
        f'<div class="label">{html.escape(k)}</div></div>'
        for k, v in {**{"OVERALL": summary.get("overall", 0)}, **summary.get("family_means", {})}.items()
    )

    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in SCORE_COLUMNS)

    def color(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        # green-ish for high, red-ish for low (only for [0,1] scores)
        if v <= 1.0:
            r = int(255 * (1 - v)); g = int(180 * v)
            return f' style="background:rgba({r},{g},80,0.18)"'
        return ""

    rows = ""
    for r in records:
        score_cells = ""
        for key, _ in SCORE_COLUMNS:
            v = _get(r, key)
            score_cells += f"<td{color(v)}>{_fmt(v)}</td>"
        rows += (
            f'<tr><td>{html.escape(str(r["id"]))}</td><td>{html.escape(r.get("type",""))}</td>'
            f'<td class="qa">{html.escape(r["question"])}</td>'
            f'<td class="qa">{html.escape(r["reference_answer"])}</td>'
            f'<td class="qa">{html.escape(r["rag_answer"])}</td>'
            f"{score_cells}</tr>"
        )

    # Aggregate row (means of normalized columns; raw judge means shown too).
    agg_cells = ""
    for key, _ in SCORE_COLUMNS:
        vals = [_get(r, key) for r in records]
        vals = [v for v in vals if isinstance(v, (int, float))]
        agg_cells += f"<td><b>{(sum(vals)/len(vals)):.3f}</b></td>" if vals else "<td></td>"
    agg_row = f'<tr class="agg"><td colspan="5"><b>AGGREGATE (mean of {len(records)})</b></td>{agg_cells}</tr>'

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>RGD RAG Evaluation</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f5f5;color:#222;margin:0;padding:20px}}
h1{{color:#1a237e}} .stats{{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}}
.stat-card{{background:#fff;border-radius:8px;padding:12px 18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.12)}}
.stat-card .number{{font-size:1.5em;font-weight:700;color:#1a237e}} .stat-card .label{{font-size:.75em;color:#666}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:.8em}}
th,td{{border:1px solid #e0e0e0;padding:6px 8px;text-align:center}}
th{{background:#1a237e;color:#fff;position:sticky;top:0}}
td.qa{{text-align:left;max-width:280px;font-size:.95em;vertical-align:top}}
tr.agg td{{background:#fff8e1}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>RGD RAG Evaluation</h1>
<div class="stats">{cards}</div>
<div class="wrap"><table>
<thead><tr><th>ID</th><th>Type</th><th>Question</th><th>Reference A</th><th>RAG A</th>{head}</tr></thead>
<tbody>{agg_row}{rows}</tbody>
</table></div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def render_report(records: List[Dict], summary: Dict, out_dir: str):
    write_csv(records, os.path.join(out_dir, "eval_results.csv"))
    write_html(records, summary, os.path.join(out_dir, "eval_report.html"))
    write_markdown(records, summary, os.path.join(out_dir, "eval_report.md"))
