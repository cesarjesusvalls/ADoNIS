"""Figures-only validation document: every §1 ADoNIS-vs-ACHILLES validation panel, embedded, no prose.
T2K (carbon) + cross-experiment (MINERvA flux on C, MicroBooNE flux on Ar).

Usage:  python -m analysis.paper.validation_figs   ->  output/paper/validation_figs.html
"""
import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
DEST = OUT / "paper" / "validation_figs.html"
MAXW = 1200

# (file, one-line title) in reading order; grouped by section header (None = figure)
ITEMS = [
    ("H", "T2K flux · ¹²C  —  the primary validation"),
    ("paper/sec1_cc0pi.png",        "CC0π kinematics (δp_T, δα_T, p_μ, cosθ_μ)"),
    ("paper/sec1_cc1pi.png",        "CC1π⁺ STV (p_N, δp_TT, δα_T, p_π, cosθ_π)"),
    ("paper/sec1_multiplicity.png", "CC-inclusive multiplicities (N_p, N_π±)"),
    ("paper/beams_validation.png",  "Tagged beams π⁺/p/n on ¹²C (pure transport)"),
    ("H", "MINERvA flux (NuMI ~4 GeV) · ¹²C  —  different flux"),
    ("paper/xval_minerva_stv.png",  "CC0π STV (δp_T, δα_T)"),
    ("paper/xval_minerva_muon.png", "CC0π muon kinematics (p_μ, cosθ_μ)"),
    ("H", "MicroBooNE flux (BNB ~0.8 GeV) · ⁴⁰Ar  —  different target"),
    ("paper/xval_uboone_incl.png",  "CC-inclusive muon kinematics (p_μ, cosθ_μ)"),
    ("paper/xval_uboone_cc0pi.png", "CC0πNp (δp_T, p_μ)"),
    ("H", "Fully-inclusive cross-section ratio (no cuts)"),
    ("paper/xval_inclusive_ratio.png", "Total CC σ, ADoNIS/ACHILLES vs E_ν — C and Ar"),
]


def embed(rel):
    p = OUT / rel
    if not p.exists():
        return f'<div class="miss">missing: {rel}</div>'
    im = Image.open(p).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return f'<img alt="{Path(rel).stem}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


CSS = """
:root{--bg:#f6f8fa;--panel:#fff;--ink:#1a2430;--mut:#5a6b7a;--line:#e2e8ef;--acc:#1f77b4}
@media(prefers-color-scheme:dark){:root{--bg:#0d1218;--panel:#151d27;--ink:#e6edf3;--mut:#9fb0c0;--line:#243040;--acc:#4a9fd8}}
:root[data-theme=light]{--bg:#f6f8fa;--panel:#fff;--ink:#1a2430;--mut:#5a6b7a;--line:#e2e8ef;--acc:#1f77b4}
:root[data-theme=dark]{--bg:#0d1218;--panel:#151d27;--ink:#e6edf3;--mut:#9fb0c0;--line:#243040;--acc:#4a9fd8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:25px;margin:0 0 4px}.sub{color:var(--mut);font-size:14px;margin:0 0 8px}
h2{font-size:17px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--line);color:var(--acc)}
figure{margin:16px 0 6px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
figure img{display:block;width:100%;height:auto;border-radius:4px}
figcaption{color:var(--mut);font-size:12.5px;margin-top:8px}
.miss{color:#b9770a;font-size:13px}
"""


def main():
    body = ['<div class="wrap">',
            "<h1>ADoNIS vs ACHILLES — validation figures</h1>",
            '<p class="sub">Every §1 validation panel. ADoNIS (blue squares) on the ACHILLES step + stat band; '
            'ratio ACH/ADO on the ±5% band. Percent-level agreement across three fluxes and two targets.</p>']
    for a, b in ITEMS:
        if a == "H":
            body.append(f"<h2>{b}</h2>")
        else:
            body.append(f"<figure>{embed(a)}<figcaption>{b}</figcaption></figure>")
    body.append("</div>")
    html = f"<!doctype html><meta charset=utf-8><title>ADoNIS validation figures</title><style>{CSS}</style>" + "".join(body)
    DEST.write_text(html)
    print(f"[out] {DEST}  ({DEST.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
