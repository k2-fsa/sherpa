#!/usr/bin/env python3

from pathlib import Path
from jinja2 import Template

import os


def get_sherpa_onnx_version():
    return os.environ.get("SHERPA_ONNX_VERSION", "1.12.23")


def main():

    # Input Jinja2 template
    template_path = Path("tpl/download-pre-compiled.rst")

    # Output directory
    output_dir = Path("generated/download")
    output_dir.mkdir(exist_ok=True)

    # Supported Windows architectures
    ARCHS = {
        "windows_x64": "x64",
        "windows_x86": "x86",
        "windows_arm64": "arm64",
    }

    template_text = template_path.read_text(encoding="utf-8")
    template = Template(template_text)

    for platform_name, arch in ARCHS.items():
        rendered = template.render(
            sherpa_onnx_version=get_sherpa_onnx_version(),
            arch=arch,
        )

        output_file = output_dir / f"{platform_name}.rst"
        output_file.write_text(rendered, encoding="utf-8")

        print(f"Generated {output_file}")


if __name__ == "__main__":
    main()
