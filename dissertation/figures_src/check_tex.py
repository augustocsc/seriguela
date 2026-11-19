"""Consistency checks for the revised dissertation LaTeX sources."""
import re
import glob
import os
from collections import Counter

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(BASE)

texs = sorted(glob.glob("chapters/*.tex")) + ["ic-tese-v3.tex"]
allsrc = ""
for t in texs:
    allsrc += open(t, encoding="utf-8").read()

# 1) citations vs bib
bib = open("bibliography/ic-tese-v3.bib", encoding="utf-8").read()
bibkeys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
cites = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", allsrc):
    for k in m.group(1).split(","):
        cites.add(k.strip())
missing = sorted(cites - bibkeys)
print("cited keys:", len(cites), "| missing from bib:", missing)
print("bib entries never cited:", len(bibkeys - cites))

# 2) figures
figs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", allsrc))
bad = [f for f in sorted(figs) if not os.path.exists(f)]
print("figures referenced:", len(figs), "| missing:", bad)

# 3) refs vs labels
labels = set(re.findall(r"\\label\{([^}]+)\}", allsrc))
refs = set(re.findall(r"\\ref\{([^}]+)\}", allsrc))
print("refs missing labels:", sorted(refs - labels))

# 4) environment balance per file
for t in texs:
    src = open(t, encoding="utf-8").read()
    src_nc = re.sub(r"(?<!\\)%.*", "", src)
    cb = Counter(re.findall(r"\\begin\{(\w+\*?)\}", src_nc))
    ce = Counter(re.findall(r"\\end\{(\w+\*?)\}", src_nc))
    if cb != ce:
        diff = {k: (cb.get(k, 0), ce.get(k, 0)) for k in set(cb) | set(ce)
                if cb.get(k, 0) != ce.get(k, 0)}
        print("ENV MISMATCH", t, diff)

# 5) rough brace balance per file
for t in texs:
    src = open(t, encoding="utf-8").read()
    src2 = re.sub(r"(?<!\\)%.*", "", src)
    src2 = src2.replace(r"\{", "").replace(r"\}", "")
    b = src2.count("{") - src2.count("}")
    if b != 0:
        print("BRACE IMBALANCE", t, b)

# 6) fictitious markers inventory
for t in texs:
    src = open(t, encoding="utf-8").read()
    n_fic = len(re.findall(r"\\fic\{", src))
    n_com = len(re.findall(r"FICTITIOUS", src))
    if n_fic or n_com:
        print(f"fic markers in {t}: \\fic x{n_fic}, comments x{n_com}")

print("checks done")
