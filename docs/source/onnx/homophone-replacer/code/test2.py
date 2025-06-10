import pynini
from pynini import cdrewrite
from pynini.lib import byte, utf8

sigma = utf8.VALID_UTF8_CHAR.star

rule1 = pynini.cross("xuan2jie4xin1pian4", "玄戒芯片")

# 针对前鼻音和后鼻音不分的情况
#
# 注意：可以指定多个规则，都替换成同一个词组
rule2 = pynini.cross("xuan2jie4xing1pian4", "玄戒芯片")

rule3 = pynini.cross("fu2nan2ren2", "湖南人")

rule4 = pynini.cross("gong1tou2an1zhuang1", "弓头安装")

rule5 = pynini.cross("ji1zai3chuan2gan3qi4", "机载传感器")

# 可以指定多个规则，覆盖可能的发音
rule6 = pynini.cross("ji1zai4chuan2gan3qi4", "机载传感器")

# 本例子只有6条规则。你可以添加任意多条规则。
rule = (rule1 | rule2 | rule3 | rule4 | rule5 | rule6).optimize()
rule = cdrewrite(rule, "", "", sigma)

rule.write("replace.fst")
