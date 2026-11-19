#!/usr/bin/env python3
"""Compile the dissertation PDF via WSL LaTeX."""
import subprocess
import sys
import os
from pathlib import Path

DISS_DIR = Path(r"C:\Users\Rudá\seriguela\dissertation_final")
MAIN_TEX = "ic-tese-v3.tex"

def run_wsl_cmd(cmd):
    """Run a command in WSL and return output."""
    wsl_cmd = f"wsl bash -c '{cmd}'"
    result = subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def compile_latex():
    """Compile LaTeX to PDF using pdflatex + bibtex."""
    os.chdir(DISS_DIR)

    # Check if pdflatex is available in WSL
    code, out, err = run_wsl_cmd("which pdflatex")
    if code != 0:
        print("❌ pdflatex not found in WSL. Install with:")
        print("   wsl -u root bash -c 'apt-get update && apt-get install -y texlive-latex-extra'")
        return False

    print("✓ pdflatex found in WSL")

    # Copy project to /tmp for faster access
    print("📦 Copying project to WSL /tmp...")
    code, _, _ = run_wsl_cmd(f"mkdir -p /tmp/diss && cp -r {DISS_DIR} /tmp/diss/")
    if code != 0:
        print("⚠️  Copying with absolute path failed, trying relative...")

    # Run compilation in WSL
    wsl_work_dir = "/mnt/c/Users/Rudá/seriguela/dissertation_final"

    # First pass: pdflatex
    print("📄 First pass: pdflatex...")
    code, out, err = run_wsl_cmd(
        f"cd '{wsl_work_dir}' && pdflatex -interaction=nonstopmode {MAIN_TEX} 2>&1 | tail -20"
    )
    if "Fatal error" in out or "Fatal error" in err:
        print(f"❌ pdflatex failed:\n{out}\n{err}")
        return False
    print(out)

    # bibtex pass
    print("\n📚 Running bibtex for references...")
    base = MAIN_TEX.replace(".tex", "")
    code, out, err = run_wsl_cmd(f"cd '{wsl_work_dir}' && bibtex {base} 2>&1")
    print(out)

    # Second pass: pdflatex
    print("\n📄 Second pass: pdflatex...")
    code, out, err = run_wsl_cmd(
        f"cd '{wsl_work_dir}' && pdflatex -interaction=nonstopmode {MAIN_TEX} 2>&1 | tail -20"
    )
    print(out)

    # Third pass: pdflatex (for cross-references)
    print("\n📄 Third pass: pdflatex...")
    code, out, err = run_wsl_cmd(
        f"cd '{wsl_work_dir}' && pdflatex -interaction=nonstopmode {MAIN_TEX} 2>&1 | tail -20"
    )
    print(out)

    # Check if PDF was created
    pdf = DISS_DIR / f"{MAIN_TEX.replace('.tex', '.pdf')}"
    if pdf.exists():
        size_mb = pdf.stat().st_size / (1024*1024)
        print(f"\n✅ PDF compiled successfully: {pdf.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"\n❌ PDF not found at {pdf}")
        return False

if __name__ == "__main__":
    success = compile_latex()
    sys.exit(0 if success else 1)
