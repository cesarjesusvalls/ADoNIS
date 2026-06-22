"""Assemble all CC matrix figures (paper_figures/matrix/*.png) into one PDF, one figure per page,
ordered material -> channel -> contribution -> proton.  Usage: python scripts/build_matrix_pdf.py [out.pdf]"""
import os, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper_figures/cc_matrix_all.pdf"
D = sys.argv[2] if len(sys.argv) > 2 else "paper_figures/matrix"
order_mat = ["C", "Ar"]; order_ch = ["cc0pi", "cc1pi"]; order_co = ["qe", "res", "both"]; order_p = ["incl", "0p", "1p", "2p"]
def key(fn):
    b = os.path.basename(fn)[:-4]; parts = b.split("_")  # channel_contrib_pcat_material
    ch, co, pc, mat = parts[0], parts[1], parts[2], parts[3]
    def idx(L, x): return L.index(x) if x in L else 99
    return (idx(order_mat, mat), idx(order_ch, ch), idx(order_co, co), idx(order_p, pc))
figs = sorted(glob.glob(os.path.join(D, "*.png")), key=key)
print(f"{len(figs)} figures -> {OUT}", flush=True)
with PdfPages(OUT) as pdf:
    for fn in figs:
        img = mpimg.imread(fn); h, w = img.shape[:2]
        fig = plt.figure(figsize=(min(11, w/100), min(8.5, h/100)))
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
        pdf.savefig(fig, dpi=150); plt.close(fig)
print("PDF DONE", flush=True)
