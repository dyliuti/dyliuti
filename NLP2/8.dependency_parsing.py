from pyhanlp import *

sentence = "徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"
print(HanLP.parseDependency(sentence))

f = open("result.txt", 'a+')
print((HanLP.parseDependency(sentence)), file=f)
