#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Input directory.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )

    return parser.parse_args()


def process(out_dir: Path, whl: Path):
    tmp_dir = out_dir / "tmp"
    subprocess.check_call(f"unzip {whl} -d {tmp_dir}", shell=True)
    py_version = ".".join(sys.version.split(".")[:2])
    rpath_list = [
        f"$ORIGIN/../lib/python{py_version}/site-packages/k2_sherpa.libs",
        f"$ORIGIN/../lib/python{py_version}/site-packages/torch/lib",
        f"$ORIGIN/../lib/python{py_version}/site-packages/torch/lib64",
        f"$ORIGIN/../lib/python{py_version}/site-packages/k2/lib",
        f"$ORIGIN/../lib/python{py_version}/site-packages/k2/lib64",
        f"$ORIGIN/../lib/python{py_version}/site-packages/kaldifeat/lib",
        f"$ORIGIN/../lib/python{py_version}/site-packages/kaldifeat/lib64",
        #
        f"$ORIGIN/../lib/python{py_version}/dist-packages/k2_sherpa.libs",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/torch/lib",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/torch/lib64",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/k2/lib",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/k2/lib64",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/kaldifeat/lib",
        f"$ORIGIN/../lib/python{py_version}/dist-packages/kaldifeat/lib64",
    ]
    rpaths = ":".join(rpath_list)

    for filename in glob.glob(f"{tmp_dir}/k2_sherpa-*data/data/bin/*", recursive=True):
        print(filename)
        existing_rpath = (
            subprocess.check_output(["patchelf", "--print-rpath", filename])
            .decode()
            .strip()
        )
        target_rpaths = rpaths + ":" + existing_rpath
        subprocess.check_call(
            f"patchelf --force-rpath --set-rpath '{target_rpaths}' {filename}",
            shell=True,
        )

    outwheel = Path(shutil.make_archive(whl, "zip", tmp_dir))
    Path(outwheel).rename(out_dir / whl.name)

    shutil.rmtree(tmp_dir)


def main():
    in_dir = get_args().in_dir
    out_dir = get_args().out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    for whl in in_dir.glob("*.whl"):
        process(out_dir, whl)


if __name__ == "__main__":
    main()
