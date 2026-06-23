"""CASCADE-VERTEX MATRIX: ADoNIS vs ACHILLES primary first-interaction comparison -- the cascade
co-diagnostic to the cross-section matrix (gen_cc_matrix).  For each PRIMARY cascade particle, both
generators record its FIRST interaction (channel + products) at NOMINAL; this driver compares, per
production process (QE/RES) x incident-particle type x channel:
  (1) the interaction-channel FRACTION vs incident |p|  (chi2/pull, the primary metric)
  (2) the per-channel PRODUCT |p| spectrum, per produced-particle type
Weighting is apples-to-apples: ADoNIS primaries are importance-sampled (weight w); ACHILLES events
carry the physical per-event weight (VTX w=..).  Fractions are weight-normalized within each (proc,
type, |p|) cell so they estimate the same flux-averaged conditional cascade fate on both sides.

ADoNIS input : cascade_vertex_<mat>_ado.npz  (gen_cascade_vertex.py)
ACHILLES input: cascade_vertex_<mat>_ach.txt  (achilles:vertex, ACHILLES_VERTEXDUMP=1, the `VTX ...` lines)

Run: python -u scripts/cascade_vertex_matrix.py <mat> <ado.npz> <ach.txt> [out_prefix]
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

argv = [a for a in sys.argv[1:] if a != "--primaries"]
PRIMARIES = "--primaries" in sys.argv                            # filter is_first==1 -> the primary matrix
MAT = argv[0] if len(argv) > 0 else "C"
ADO = argv[1] if len(argv) > 1 else f"/tmp/cascade_segments_{MAT}_ado.npz"
ACH = argv[2] if len(argv) > 2 else f"/tmp/cascade_segments_{MAT}_ach.txt"
OUTP = (argv[3] if len(argv) > 3 else f"/tmp/cascade_segments_{MAT}") + ("_prim" if PRIMARIES else "_all")

PNAME = {211: "pi+", 111: "pi0", -211: "pi-", 2212: "p", 2112: "n"}
PIONS = (211, 111, -211); NUCS = (2212, 2112)
CH_PI = {0: "transmit", 1: "elastic", 2: "charge-ex", 3: "absorption", 4: "conversion"}
CH_NU = {0: "transmit", 1: "elastic", 2: "inelastic"}
PROC = {0: "QE", 1: "RES"}
ED = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1250, 1500, 2000, 3000.0])


def chans(pid):
    return CH_PI if pid in PIONS else CH_NU


# ---------- loaders -> common record dict of arrays (one row per cascade SEGMENT) ----------
def load_ado(path):
    d = np.load(path, allow_pickle=True)
    return dict(w=d["w"], proc=d["proc"], is_first=d["is_first"].astype(bool), inc_pid=d["inc_pid"],
                inc_p=d["inc_p"], channel=d["channel"], prod_pid=d["prod_pid"], prod_p=d["prod_p"],
                nprod=d["nprod"])


_VTX = re.compile(r"VTX w=(\S+) proc=(\d+) is_primary=(\d+) is_first=(\d+) inc_pid=(-?\d+) inc_p=(\S+) channel=(\d+) nprod=(\d+)(.*)")
_PRD = re.compile(r"pid=(-?\d+) p=(\S+)")


def load_ach(path):
    w, proc, isf, ipid, ip, ch, nprod = [], [], [], [], [], [], []
    ppid, pp = [], []                                            # ragged -> padded to 3
    with open(path) as fh:
        for line in fh:
            if not line.startswith("VTX "):
                continue
            m = _VTX.match(line)
            if not m:
                continue
            w.append(float(m.group(1))); proc.append(int(m.group(2))); isf.append(int(m.group(4)))
            ipid.append(int(m.group(5))); ip.append(float(m.group(6))); ch.append(int(m.group(7)))
            nprod.append(int(m.group(8)))
            prods = _PRD.findall(m.group(9))
            row_pid = [0, 0, 0]; row_p = [0.0, 0.0, 0.0]
            for k, (pid, pmag) in enumerate(prods[:3]):
                row_pid[k] = int(pid); row_p[k] = float(pmag)
            ppid.append(row_pid); pp.append(row_p)
    return dict(w=np.array(w), proc=np.array(proc), is_first=np.array(isf, bool), inc_pid=np.array(ipid),
                inc_p=np.array(ip), channel=np.array(ch), prod_pid=np.array(ppid),
                prod_p=np.array(pp), nprod=np.array(nprod))


def sel(D, proc, pid):
    """segments of a given process + incident pid.  With --primaries, restrict to is_first==1 (the FIRST
    segment of an original primary) -> recovers the primary matrix exactly; else ALL cascade segments."""
    m = (D["proc"] == proc) & (D["inc_pid"] == pid)
    if PRIMARIES:
        m &= D["is_first"]
    return m


def wfrac(w, mask_bracket, mask_chan):
    """weighted channel fraction within a bracket + binomial error from effective N."""
    wb = w[mask_bracket]
    if wb.sum() <= 0:
        return np.nan, np.nan, 0.0
    f = (w[mask_bracket & mask_chan]).sum() / wb.sum()
    neff = wb.sum() ** 2 / np.clip((wb ** 2).sum(), 1e-300, None)
    err = np.sqrt(max(f * (1 - f), 1e-9) / max(neff, 1.0))
    return f, err, neff


# ---------- channel-fraction matrix ----------
def fraction_matrix(A, B, log):
    """A=ACHILLES, B=ADoNIS.  Per (proc, inc pid): fraction vs |p| bracket, ratio + pull + chi2/ndf."""
    rows = []
    for proc in (0, 1):
        for pid in (PIONS + NUCS):
            mA = sel(A, proc, pid); mB = sel(B, proc, pid)
            if A["w"][mA].sum() <= 0 and B["w"][mB].sum() <= 0:
                continue
            CH = chans(pid)
            log.append(f"\n=== {PROC[proc]}  incident {PNAME[pid]}   (ACH N={int(mA.sum())}  ADO N={int(mB.sum())}) ===")
            for c, cname in CH.items():
                chi2 = 0.0; ndf = 0; line_any = False
                cells = []
                for lo, hi in zip(ED[:-1], ED[1:]):
                    bA = mA & (A["inc_p"] >= lo) & (A["inc_p"] < hi)
                    bB = mB & (B["inc_p"] >= lo) & (B["inc_p"] < hi)
                    fA, eA, nA = wfrac(A["w"], bA, A["channel"] == c)
                    fB, eB, nB = wfrac(B["w"], bB, B["channel"] == c)
                    if not (np.isfinite(fA) and np.isfinite(fB)) or (nA < 20 and nB < 20):
                        continue
                    if fA < 1e-4 and fB < 1e-4:
                        continue
                    pull = (fA - fB) / np.sqrt(eA ** 2 + eB ** 2 + 1e-12)
                    ratio = fA / fB if fB > 0 else np.inf
                    chi2 += pull ** 2; ndf += 1; line_any = True
                    cells.append((f"[{lo:.0f},{hi:.0f})", fA, eA, fB, eB, ratio, pull))
                if line_any:
                    log.append(f"  channel {c} ({cname}):  chi2/ndf = {chi2:.1f}/{ndf} = {chi2/max(ndf,1):.2f}")
                    for nm, fA, eA, fB, eB, r, p in cells:
                        flag = "  <<" if abs(p) > 2 else ""
                        log.append(f"    {nm:>12}  ACH {fA:.4f}+-{eA:.4f}  ADO {fB:.4f}+-{eB:.4f}  "
                                   f"ratio {r:5.3f}  pull {p:+5.1f}{flag}")
                    rows.append((proc, pid, c, cname, chi2, ndf))
    return rows


# ---------- product spectra (per proc, channel, produced-type) ----------
def product_spectra(A, B, log, out_prefix):
    PB = np.array([0, 100, 200, 300, 400, 500, 700, 1000, 1500.0])
    for proc in (0, 1):
        for ipid in (PIONS + NUCS):
            CH = chans(ipid)
            for c in CH:
                if c == 0:
                    continue
                mA = sel(A, proc, ipid) & (A["channel"] == c)
                mB = sel(B, proc, ipid) & (B["channel"] == c)
                if A["w"][mA].sum() <= 0 and B["w"][mB].sum() <= 0:
                    continue
                prods = set()
                for D, m in ((A, mA), (B, mB)):
                    pp = D["prod_pid"][m]
                    for v in np.unique(pp):
                        if v != 0:
                            prods.add(int(v))
                for tp in sorted(prods):
                    hA = _spec(A, mA, tp, PB); hB = _spec(B, mB, tp, PB)
                    if hA.sum() <= 0 and hB.sum() <= 0:
                        continue
                    chi2, ndf = _spec_chi2(A, mA, B, mB, tp, PB)
                    log.append(f"  spectrum {PROC[proc]} {PNAME[ipid]}->ch{c} produced {PNAME.get(tp,tp)}: "
                               f"<NA={A['w'][mA].sum():.2e} NB={B['w'][mB].sum():.2e}>  rateA/B="
                               f"{_rate(A,mA,tp)/max(_rate(B,mB,tp),1e-30):.3f}  shapeChi2/ndf={chi2:.1f}/{ndf}")


def _rate(D, m, tp):                                            # per-primary expected count of product type tp
    pid = D["prod_pid"][m]; w = D["w"][m]
    cnt = (pid == tp).sum(1)
    return (w * cnt).sum() / max(w.sum(), 1e-30)


def _spec(D, m, tp, PB):
    pid = D["prod_pid"][m]; pp = D["prod_p"][m]; w = D["w"][m]
    sel_p = pp[pid == tp]; sel_w = np.repeat(w, (pid == tp).sum(1))
    h, _ = np.histogram(sel_p, bins=PB, weights=sel_w)
    return h


def _spec_chi2(A, mA, B, mB, tp, PB):
    hA = _spec(A, mA, tp, PB); hB = _spec(B, mB, tp, PB)
    sA, sB = hA.sum(), hB.sum()
    if sA <= 0 or sB <= 0:
        return 0.0, 0
    nA = hA / sA; nB = hB / sB                                  # SHAPE comparison (normalized)
    eA = np.sqrt(np.clip(hA, 1, None)) / sA; eB = np.sqrt(np.clip(hB, 1, None)) / sB
    d = (nA - nB) ** 2 / (eA ** 2 + eB ** 2 + 1e-12)
    use = (hA + hB) > 0
    return float(d[use].sum()), int(use.sum())


_COLLABEL = {0: "transmit", 1: "elastic", 2: "cex/inel", 3: "absorp", 4: "convert"}
# incident-type rows, in a fixed display order (proc, pid)
_ROWS = [(0, 2212), (0, 2112), (1, 211), (1, 111), (1, -211), (1, 2212), (1, 2112)]


def fig_chi2_matrix(rows, out):
    """The MATRIX view: a chi2/ndf heatmap, rows = (proc, incident type), cols = channel.  White ~1,
    red >2 (tension), grey = empty.  Each cell annotated 'chi2/ndf (ndf)'.  Analog of the xsec matrix."""
    import matplotlib.colors as mcolors
    d = {(p, pid, c): (chi2 / max(ndf, 1), ndf) for (p, pid, c, _cn, chi2, ndf) in rows}
    rlabels = [f"{PROC[p]} {PNAME[pid]}" for (p, pid) in _ROWS]
    cols = [0, 1, 2, 3, 4]
    M = np.full((len(_ROWS), len(cols)), np.nan)
    for i, (p, pid) in enumerate(_ROWS):
        for j, c in enumerate(cols):
            if (p, pid, c) in d:
                M[i, j] = d[(p, pid, c)][0]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.cm.RdYlGn_r.copy(); cmap.set_bad("0.85")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=3.0, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([_COLLABEL[c] for c in cols])
    ax.set_yticks(range(len(_ROWS))); ax.set_yticklabels(rlabels)
    ax.set_title(f"Cascade-vertex matrix  chi2/ndf  ({MAT}, {'PRIMARIES' if PRIMARIES else 'ALL SEGMENTS'})")
    for i in range(len(_ROWS)):
        for j, c in enumerate(cols):
            if not np.isnan(M[i, j]):
                v, ndf = d[(_ROWS[i][0], _ROWS[i][1], c)]
                ax.text(j, i, f"{v:.2f}\n({ndf})", ha="center", va="center", fontsize=8,
                        color="white" if v > 2.0 else "black")
    plt.colorbar(im, ax=ax, label="chi2/ndf"); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_fractions(A, B, out):
    """Per incident-type: TOP = channel fractions vs |p| (ACH solid / ADO dash), BOTTOM = ratio ADO/ACH
    per channel (+-10% guides) with the per-channel chi2/ndf in the legend.  (xsec-matrix overlay+ratio.)"""
    ctr = 0.5 * (ED[:-1] + ED[1:])
    ncol = len(_ROWS)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 7.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.3, 1]))
    for col, (proc, pid) in enumerate(_ROWS):
        at, ar = axes[0, col], axes[1, col]
        mA = sel(A, proc, pid); mB = sel(B, proc, pid)
        if A["w"][mA].sum() <= 0 and B["w"][mB].sum() <= 0:
            at.set_visible(False); ar.set_visible(False); continue
        for c, cname in chans(pid).items():
            fA = np.full(len(ctr), np.nan); fB = np.full(len(ctr), np.nan); chi2 = 0.0; ndf = 0
            for k, (lo, hi) in enumerate(zip(ED[:-1], ED[1:])):
                fa, ea, na = wfrac(A["w"], mA & (A["inc_p"] >= lo) & (A["inc_p"] < hi), A["channel"] == c)
                fb, eb, nb = wfrac(B["w"], mB & (B["inc_p"] >= lo) & (B["inc_p"] < hi), B["channel"] == c)
                if na >= 20:
                    fA[k] = fa
                if nb >= 20:
                    fB[k] = fb
                if na >= 20 and nb >= 20 and (fa > 1e-4 or fb > 1e-4):
                    chi2 += (fa - fb) ** 2 / (ea ** 2 + eb ** 2 + 1e-12); ndf += 1
            if ndf == 0:
                continue
            l, = at.plot(ctr, fA, "-", lw=1.5)
            at.plot(ctr, fB, "--", lw=1.5, color=l.get_color())
            with np.errstate(invalid="ignore", divide="ignore"):
                ar.plot(ctr, fA / fB, "-", lw=1.3, color=l.get_color(),
                        label=f"{cname} {chi2/max(ndf,1):.1f}")
        at.set_title(f"{PROC[proc]} {PNAME[pid]}", fontsize=10); at.set_ylim(0, 1); at.grid(alpha=0.3)
        if col == 0:
            at.set_ylabel("channel fraction\n(solid ACH / dash ADO)")
            ar.set_ylabel("ADO/ACH")
        ar.axhline(1, color="k", lw=0.7); ar.axhline(1.1, color="grey", ls=":"); ar.axhline(0.9, color="grey", ls=":")
        ar.set_ylim(0.6, 1.4); ar.grid(alpha=0.3); ar.set_xlabel("|p| [MeV]")
        ar.legend(fontsize=6, title="ch: chi2/ndf", title_fontsize=6, ncol=1, loc="lower left")
    fig.suptitle(f"Cascade-vertex matrix  ({MAT}, {'PRIMARIES' if PRIMARIES else 'ALL SEGMENTS'})  solid ACH / dash ADO", y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    A = load_ach(ACH); B = load_ado(ADO)
    log = [f"CASCADE-VERTEX MATRIX  material={MAT}  mode={'PRIMARIES (is_first)' if PRIMARIES else 'ALL SEGMENTS'}",
           f"ACHILLES: {ACH}  ({len(A['w'])} primaries, Sw={A['w'].sum():.3e})",
           f"ADoNIS  : {ADO}  ({len(B['w'])} primaries, Sw={B['w'].sum():.3e})"]
    log.append("\n########## CHANNEL-FRACTION MATRIX (primary metric) ##########")
    rows = fraction_matrix(A, B, log)
    log.append("\n########## PRODUCT SPECTRA (per channel, per produced type) ##########")
    product_spectra(A, B, log, OUTP)
    log.append("\n########## SUMMARY (channel chi2/ndf) ##########")
    for proc, pid, c, cname, chi2, ndf in sorted(rows, key=lambda r: -r[4] / max(r[5], 1)):
        flag = "  <<<" if chi2 / max(ndf, 1) > 2 else ""
        log.append(f"  {PROC[proc]:>3} {PNAME[pid]:>4} ch{c}({cname:<10}) chi2/ndf={chi2:6.1f}/{ndf:<3}={chi2/max(ndf,1):6.2f}{flag}")
    txt = "\n".join(log)
    print(txt)
    open(f"{OUTP}_matrix.log", "w").write(txt)
    fig_chi2_matrix(rows, f"{OUTP}_chi2matrix.png")
    fig_fractions(A, B, f"{OUTP}_fractions.png")
    print(f"\nwrote {OUTP}_matrix.log , {OUTP}_chi2matrix.png , {OUTP}_fractions.png")


if __name__ == "__main__":
    main()
