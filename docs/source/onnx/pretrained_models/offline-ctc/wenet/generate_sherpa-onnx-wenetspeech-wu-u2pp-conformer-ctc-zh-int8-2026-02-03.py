#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader

# List of WAV files and their ground truth
s = """
1.wav 而宋子文搭子宋美龄搭子端纳呢侪没经过搜查
2.wav 借搿个机会纷纷响应搿个辰光奥地利个老皇帝已经死脱了
3.wav 呃大灰狼就跟山羊奶奶讲山羊奶奶侬一家头蹲阿拉决定拿这点物事侪送拨侬
4.wav 胖胖又得意了啥人会得想到玩具汽车里头还囥了物事呢
5.wav 这物事里头是有利益分配的讲好个埃种大生意难做一趟做两三年也做不出的
6.wav 这个新生儿啊相对来讲偏少大家侪不愿意生嘛
7.wav 这自然应该是像上海大都市这能介告诉伊虽然伊同样是外来的闲话
8.wav 已经有西南亚洲的外国人居住辣辣埃及从事贸易活动
9.wav 青春的舞龙唱出短暂的曲子的清风里后世
10.wav 肠道菌群也就是阿拉肠道当中不同种类的细菌等微生物会的影响大脑的健康
11.wav 老百姓大家知了伊也勿中浪向摊头浪向吃两碗豆腐花
12.wav 孙女告娘当我儿子看我讲的闲话
13.wav 呃对伐现在实际上是新上海人越来越多了外加未来我觉着这群新上海人会得取代脱阿拉
14.wav 有搿种爷娘对伐但是我觉着现在好像就讲上海哦现在勿是侪讲房子也没人住嘛外国人跑得一批还有就是叫低生育率帮低结婚率嗯
15.wav 当侬老了一个人头发花白坐辣盖落花旁边轻轻的从书架上面取下一本书来慢慢叫的阅读
16.wav 伴着夕阳的余晖一切侪是最美好的样子
17.wav 勿晓得个呀老早勿是讲旧社会个辰光嘛搿种流氓阿了
18.wav 观众朋友们就是教个小诀窍就是屋里向大家一直拌馄饨芯子啊
19.wav 哦对的对的侬讲了对的哎哟这小米侬还是侬脑子好
20.wav 嗯沿海各地包括㑚南翔连是日本海的前头一个费城
21.wav 侬就没命了为了不叫类似的事体再发生张晨
22.wav 其实这两年我也就是行尸走肉因为老婆没了
23.wav 对的呀末伊拉这评论里向有种侬要讲一个人真个红了对勿啦就讲侬粉丝超过一万了嘛侬这种黑粉丝多
24.wav 正常保养电池呃电瓶啊搿种轮胎啊还有
"""

lines = s.strip().split("\n")
wav_list = [tuple(line.split()) for line in lines]

model = "sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03"


# Convert to dicts for Jinja2
wav_files = [
    {
        "filename": f,
        "ground_truth": g,
        "audio_src": f"/sherpa/_static/{model}/{f}",
    }
    for f, g in wav_list
]

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
