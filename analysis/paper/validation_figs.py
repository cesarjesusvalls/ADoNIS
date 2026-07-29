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

# (file, one-line title) in reading order; grouped by section header (H = section header).
# The ACHILLES-paper (arXiv:2508.19213) reproduction set built by sec1_validation/make.py.
ITEMS = [
    ("H", "Inclusive & free-nucleon / pion-nucleus cross sections"),
    ("paper/fig01_ee_domega.png",       "Fig 1 — inclusive (e,e') dσ/dω, ⁴⁰Ar & ¹²C @2.222 GeV (QE/RES/total)"),
    ("paper/fig02_anl_sigma.png",       "Fig 2 — free-nucleon RES single-pion σ(E_ν), 3 CC channels"),
    ("paper/fig03_pi_nucleus_sigma.png", "Fig 3 — π⁺ nucleus absorption + reaction σ(p), ¹²C & ⁴⁰Ar"),
    ("paper/fig13_piN_sigma.png",       "Fig 13 — meson-baryon DCC σ(W) + angular (ADoNIS MC vs ANL-Osaka)"),
    ("H", "e4ν electron scattering · ¹²C @1.159 GeV"),
    ("paper/fig456_e4nu.png",           "Figs 4/5/6 — e4ν (e,e'): E_QE (0π), E_cal (1p0π), P_T (1p0π)"),
    ("H", "Neutrino transverse-kinematics (TKI/STV)"),
    ("paper/fig07_t2k_cc0pi.png",       "Fig 7 — T2K CC0π TKI (δp_T, δα_T, …) on ¹²C"),
    ("paper/fig08_t2k_cc1pi.png",       "Fig 8 — T2K CC1π⁺ STV (p_N, δp_TT, …) on ¹²C"),
    ("paper/fig09_minerva_cc0pi.png",   "Fig 9 — MINERvA CC0π TKI (δα_T, p_n^reco, …) on ¹²C"),
    ("paper/fig10_uboone_cc1p0pi.png",  "Fig 10 — MicroBooNE CC1p0π δp_T in δα_T slices on ⁴⁰Ar"),
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
