import sys
import subprocess
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def _run_cmdline(infile: Path, points, locations):
    cmd = f"plotdigitizer {str(infile)} "
    pts = " ".join([f"-p {','.join(map(str,pt))}" for pt in points])
    locs = " ".join([f"-l {','.join(map(str,pt))}" for pt in locations])
    cmd += f"{pts} {locs}"
    outfile = infile.with_suffix(".result.png")
    trajfile = infile.with_suffix(".result.csv")
    cmd += f" --plot {str(outfile)} --output {trajfile}"
    r = subprocess.run(cmd)
    print(points)
    return trajfile

def _check_csv_file(csvfile):
    data = np.loadtxt(csvfile)
    y = data[:, 1]
    assert y.std() > 0.0
    assert y.min() < y.mean() < y.max()


def test_trimmeed():
    csvfile = _run_cmdline(
        HERE / "img1.PNG",
        [(0, 0), (20, 0), (0, 1)],
        [(22, 26), (142, 23), (23, 106)],
    )
    _check_csv_file(csvfile)


test_trimmeed()