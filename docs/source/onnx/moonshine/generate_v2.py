#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader


def get_lang(m):
    code = m.split("-")[-1]
    code2lang = {
        "ko": "Korea",
        "ja": "Japanese",
        "en": "English",
        "zh": "Chinese",
        "vi": "Vietnamese",
        "uk": "Ukrainian",
        "ja": "Japanese",
        "es": "Spanish",
        "ar": "Arabic",
    }
    return code2lang[code]


def get_name(m):
    return f"sherpa-onnx-moonshine-{m}-quantized-2026-02-27"


def get_apk_name(m):
    return m.replace("-", "_")


def get_url(m):
    return f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-{m}-quantized-2026-02-27.tar.bz2"


def main():
    models = [
        "tiny-ko",
        "tiny-ja",
        "tiny-en",
        "base-zh",
        "base-vi",
        "base-uk",
        "base-ja",
        "base-es",
        "base-en",
        "base-ar",
    ]

    for m in models:
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template(f"./tpl/model_v2.rst")
        with open(f"./code/{m}.txt") as f:
            decode_out = f.read()

        output = template.render(
            model_name=get_name(m),
            lang=get_lang(m),
            url=get_url(m),
            apk_name=get_apk_name(m),
            m=m,
            decode_out=decode_out,
        )

        # Write to file
        out_file = f"./generated/v2/{m}.rst"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"{out_file} created successfully!")


if __name__ == "__main__":
    main()
