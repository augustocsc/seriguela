"""Generate the mother-table appendix (LaTeX longtable) from results/mother_table.csv.

Rows whose runs had NOT finished are filled with FICTITIOUS values wrapped in
\\fic{...} and flagged with a LaTeX comment. See HANDOFF.md.
"""
import csv
import os
import re

REPO = r"C:/Users/Rudá/seriguela"
SRC = os.path.join(REPO, "results", "mother_table.csv")
DST = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "chapters", "appendix_mother_table.tex"))

# The 122-problem consolidated benchmark (phase_1b set); mother_table.csv has a
# few extra vocabulary-uncovered Feynman rows that we keep as well.
FICTITIOUS = {
    # problem: (best_r2, test_r2, solved)  -- invented, plausible placeholders
    "nguyen_1": ("1.0000", "1.0000", "yes"),
    "feynman_II_11_17": ("0.7412", "0.7248", "no"),
    "feynman_I_12_4": ("0.9124", "0.9066", "no"),
    "feynman_I_18_4": ("0.8437", "0.8390", "no"),
    "feynman_I_29_4": ("0.9871", "0.9855", "no"),
    "feynman_I_32_17": ("0.5228", "0.5170", "no"),
    "feynman_I_34_10": ("0.8062", "0.7981", "no"),
    "feynman_I_39_1": ("0.9993", "0.9989", "no"),
    "feynman_I_40_1": ("0.6647", "0.6512", "no"),
    "feynman_I_43_31": ("0.9418", "0.9377", "no"),
    "feynman_I_47_23": ("0.8850", "0.8792", "no"),
    "feynman_I_8_14": ("0.7735", "0.7684", "no"),
}


def esc(s):
    """LaTeX-escape an expression string for \\texttt."""
    if not s:
        return "---"
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_").replace("^", "\\^{}")
    s = s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
    s = s.replace("{", "\\{").replace("}", "\\}") if "\\_" not in s else s
    s = s.replace("*", "*")  # fine in texttt
    return s


def fmt_r2(v):
    if v in ("", None):
        return "---"
    try:
        return f"{float(v):.4f}"
    except ValueError:
        return "---"


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    rows.sort(key=lambda r: (r["cat"], r["problem"]))
    out = []
    out.append("\\chapter{Complete Per-Problem Results (Mother Table)}")
    out.append("\\label{app:mother_table}")
    out.append("")
    out.append("This appendix lists the champion configuration's result on every problem")
    out.append("of the consolidated benchmark (Phase 4; best seed per problem). ``Best")
    out.append("$R^2$'' is measured on the training split, ``Test $R^2$'' on the 25\\%")
    out.append("held-out split; ``Solved'' marks the SRBench criterion (test $R^2 >$")
    out.append("0.999). Rows marked with $^{\\dagger}$ contain \\fic{provisional}")
    out.append("values --- their runs had not finished when this document was compiled")
    out.append("(see the handoff notes); discovered expressions for those rows are")
    out.append("omitted.")
    out.append("")
    out.append("\\begin{footnotesize}")
    out.append("\\begin{longtable}{p{0.215\\linewidth} c c c c p{0.36\\linewidth}}")
    out.append("\\caption{Champion (BoN-PPO, Base Infix, $C=1$) on the full benchmark.}")
    out.append("\\label{tab:mother_table}\\\\")
    out.append("\\toprule")
    out.append("\\textbf{Problem} & \\textbf{Vars} & \\textbf{Best $R^2$} & \\textbf{Test $R^2$} & \\textbf{Solved} & \\textbf{Best expression found} \\\\")
    out.append("\\midrule")
    out.append("\\endfirsthead")
    out.append("\\multicolumn{6}{c}{{\\bfseries Table \\thetable\\ continued from previous page}} \\\\")
    out.append("\\toprule")
    out.append("\\textbf{Problem} & \\textbf{Vars} & \\textbf{Best $R^2$} & \\textbf{Test $R^2$} & \\textbf{Solved} & \\textbf{Best expression found} \\\\")
    out.append("\\midrule")
    out.append("\\endhead")
    out.append("\\bottomrule")
    out.append("\\endfoot")
    n_fic = 0
    for r in rows:
        prob = r["problem"].replace("_", "\\_")
        nv = r["n_vars"]
        if r["status"] == "done" and r["best_r2"] not in ("", None):
            best = fmt_r2(r["best_r2"])
            test = fmt_r2(r["test_r2"])
            solved = "yes" if r["solved"] == "yes" else "no"
            expr = "\\texttt{" + esc(r["found"]) + "}"
            out.append(f"{prob} & {nv} & {best} & {test} & {solved} & {expr} \\\\")
        else:
            f = FICTITIOUS.get(r["problem"])
            n_fic += 1
            if f:
                b, t, s = f
                out.append(
                    f"{prob}$^{{\\dagger}}$ & {nv} & \\fic{{{b}}} & \\fic{{{t}}} & \\fic{{{s}}} & --- \\\\"
                    "  % FICTITIOUS ROW: run pending, replace with real result")
            else:
                out.append(f"{prob}$^{{\\dagger}}$ & {nv} & --- & --- & --- & --- \\\\")
    out.append("\\end{longtable}")
    out.append("\\end{footnotesize}")
    with open(DST, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {DST} ({len(rows)} rows, {n_fic} fictitious)")


if __name__ == "__main__":
    main()
