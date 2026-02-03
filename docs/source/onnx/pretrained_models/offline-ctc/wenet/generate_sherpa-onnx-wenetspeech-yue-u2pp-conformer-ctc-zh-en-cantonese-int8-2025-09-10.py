#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader

# List of WAV files and their ground truth
wav_list = [
    ("yue-0.wav", "两只小企鹅都有嘢食"),
    (
        "yue-1.wav",
        "叫做诶诶直入式你个脑部里边咧记得呢一个嘅以前香港有一个广告好出名嘅佢乜嘢都冇噶净系影住喺弥敦道佢哋间铺头嘅啫但系就不停有人嗌啦平平吧平吧",
    ),
    ("yue-2.wav", "忽然从光线死角嘅阴影度窜出一只大猫"),
    ("yue-3.wav", "今日我带大家去见识一位九零后嘅靓仔咧"),
    ("yue-4.wav", "香港嘅消费市场从此不一样"),
    ("yue-5.wav", "景天谂唔到呢个守门嘅弟子竟然咁无礼霎时间面色都变埋"),
    (
        "yue-6.wav",
        "六个星期嘅课程包括六堂课同两个测验你唔掌握到基本嘅十九个声母五十六个韵母同九个声调我哋仲针对咗广东话学习者会遇到嘅大樽颈啊以国语为母语人士最难掌握嘅五大韵母教课书唔会教你嘅七种变音同十种变调说话生硬唔自然嘅根本性问题提供全新嘅学习方向等你突破难关",
    ),
    ("yue-7.wav", "同意嘅累积唔系阴同阳嘅累积可以讲三既融合咗一同意融合咗阴同阳"),
    (
        "yue-8.wav",
        "而较早前已经复航嘅氹仔北安码头星期五开始增设夜间航班不过两个码头暂时都冇凌晨班次有旅客希望尽快恢复可以留喺澳门长啲时间",
    ),
    (
        "yue-9.wav",
        "刘备仲马鞭一指蜀兵一齐掩杀过去打到吴兵大败唉刘备八路兵马以雷霆万钧之势啊杀到吴兵啊尸横遍野血流成河",
    ),
    ("yue-10.wav", "原来王力宏咧系佢家中里面咧成就最低个吓哇"),
    ("yue-11.wav", "无论你提出任何嘅要求"),
    ("yue-12.wav", "咁咁多样材料咁我哋首先第一步处理咗一件"),
    (
        "yue-13.wav",
        "啲点样对于佢哋嘅服务态度啊不透过呢一年左右嘅时间啦其实大家都静一静啦咁你就会见到香港嘅经济其实",
    ),
    (
        "yue-14.wav",
        "就即刻会同贵正两位八代长老带埋五名七代弟子前啲灵蛇岛想话生擒谢信抢咗屠龙宝刀翻嚟献俾帮主嘅",
    ),
    ("yue-15.wav", "我知道我的观众大部分都是对广东话有兴趣想学广东话的人"),
    ("yue-16.wav", "诶原来啊我哋中国人呢讲究物极必反"),
    (
        "yue-17.wav",
        "如果东边道建成咁丹东呢就会成为最近嘅出海港同埋经过哈大线出海相比绥分河则会减少运渠三百五十六公里",
    ),
]

# Convert to dicts for Jinja2
wav_files = [
    {
        "filename": f,
        "ground_truth": g,
        "audio_src": f"/sherpa/_static/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/{f}",
    }
    for f, g in wav_list
]

# Setup paths for tokens, model, test_wavs
model = "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10"

# Load template
env = Environment(loader=FileSystemLoader("."))
template = env.get_template(f"./tpl/{model}.rst")

# Render template
output = template.render(
    wav_files=wav_files,
    model_path=model,
)

# Write to file
out_file = f"./generated/{model}/index.rst"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(output)

print(f"{out_file} created successfully!")
