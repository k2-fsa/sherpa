#!/usr/bin/env python3
from pathlib import Path

from jinja2 import Template

from generate_download import get_sherpa_onnx_version


def main():
    # Paths
    TEMPLATE_PATH = Path("./tpl/build-cpu.rst")
    OUTPUT_DIR = Path("generated/build_cpu")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Architectures
    ARCHS = [
        {"name": "x64", "cmake": "x64"},
        {"name": "x86", "cmake": "Win32"},
        {"name": "ARM64", "cmake": "ARM64"},
    ]

    LIB_TYPES = ["shared", "static"]
    RUNTIMES = ["MD", "MT"]
    BUILD_TYPES = ["Release", "Debug", "MinSizeRel", "RelWithDebInfo"]

    # Load template
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = Template(template_text)

    # Render per architecture
    for arch_cfg in ARCHS:
        context = {
            "arch": arch_cfg["name"],
            "cmake_arch": arch_cfg["cmake"],
            "lib_types": LIB_TYPES,
            "runtimes": RUNTIMES,
            "build_types": BUILD_TYPES,
        }

        output_file = OUTPUT_DIR / f"windows_{arch_cfg['name'].lower()}_cpu_build.rst"
        output_file.write_text(template.render(**context), encoding="utf-8")
        print(f"Generated {output_file}")


if __name__ == "__main__":
    main()
