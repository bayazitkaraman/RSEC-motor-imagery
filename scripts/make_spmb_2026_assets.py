"""Regenerate current tables and two data figures using the archived builder."""
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/spmb_2026/SPMB_2026_latex_source.zip"
DATA = ROOT / "results/summary/spmb_2026"
OUTPUT = ROOT / "results/figures/spmb_2026"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE) as archive, TemporaryDirectory() as temporary:
        project = Path(temporary) / "manuscript"
        support = project / "revision_support"
        (support / "data").mkdir(parents=True)
        for path in sorted(DATA.glob("*.csv")):
            archived = archive.read("revision_support/data/" + path.name)
            if path.read_bytes() != archived:
                raise ValueError(f"CSV differs from manuscript archive: {path.name}")
            shutil.copy2(path, support / "data" / path.name)
        builder = support / "build_assets.py"
        builder.write_bytes(archive.read("revision_support/build_assets.py"))
        subprocess.run([sys.executable, str(builder)], check=True)
        for stem in ("participant_spatial_results", "comparison_uncertainty"):
            for extension in (".pdf", ".png"):
                shutil.copy2(support / (stem + extension), OUTPUT / (stem + extension))
        for name in ("table_primary.tex", "table_comparisons.tex", "table_sensitivity.tex"):
            shutil.copy2(support / name, OUTPUT / name)
        for name in ("pipeline.png", "fig3_real_minus_surrogate_delta.png"):
            (OUTPUT / name).write_bytes(archive.read(name))
    print(f"Current manuscript assets saved to {OUTPUT}")


if __name__ == "__main__":
    main()
