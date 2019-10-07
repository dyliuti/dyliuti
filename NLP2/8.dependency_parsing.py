from pyhanlp import *

sentence = "句法分析是自然语言处理中的关键技术之一，其基本任务是确定句子的句法结构或者句子中词汇之间的依存关系。"
print(HanLP.parseDependency(sentence))

# f = open("result.txt", 'a+')
# print((HanLP.parseDependency(sentence)), file=f)
