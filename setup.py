#!/usr/bin/env python3

import re
import sys
from pathlib import Path

import setuptools

from cmake.cmake_extension import BuildExtension, bdist_wheel, cmake_extension

if sys.version_info < (3,):
    # fmt: off
    print(
        "Python 2 has reached end-of-life and is no longer supported by sherpa."
    )
    # fmt: on
    sys.exit(-1)

if sys.version_info < (3, 7):
    print(
        "Python 3.6 has reached end-of-life on December 31st, 2021 "
        "and is no longer supported by sherpa."
    )
    sys.exit(-1)


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(SHERPA_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')
    return latest_version


def get_binaries_to_install():
    bin_dir = Path("build") / "sherpa" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = []
    for f in ["sherpa", "sherpa-version"]:
        t = bin_dir / f
        exe.append(str(t))
    return exe


package_name = "k2-sherpa"

with open("sherpa/python/sherpa/__init__.py", "a") as f:
    f.write(f"__version__ = '{get_package_version()}'\n")

setuptools.setup(
    name=package_name,
    version=get_package_version(),
    author="The sherpa development team",
    author_email="dpovey@gmail.com",
    package_dir={
        "sherpa": "sherpa/python/sherpa",
    },
    data_files=[("bin", get_binaries_to_install())],
    packages=["sherpa"],
    url="https://github.com/k2-fsa/sherpa",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("_sherpa")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7.0",
    license="Apache licensed, as found in the LICENSE file",
)

# remove the line __version__ from sherpa/python/sherpa/__init__.py
with open("sherpa/python/sherpa/__init__.py", "r") as f:
    lines = f.readlines()

with open("sherpa/python/sherpa/__init__.py", "w") as f:
    for line in lines:
        if "__version__" in line and "torch" not in line:
            # skip __version__ = "x.x.x"
            continue
        f.write(line)
