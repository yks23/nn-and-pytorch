# 简单小实验
## 实验零
https://playground.tensorflow.org/
不用写代码的活，试试把这个网站上的按钮都按一按。
## 实验一
本部分包含三个小实验，希望通过这些实验，帮助你对最基本的超参有所了解。
参看mnist.ipynb，运行其中的代码，观察不同超参对模型训练速度和最终性能的影响。
mnist，梦的开始。
## 实验二
参看sin.ipynb，这个实验是用一个多项式函数拟合sin(x)函数。
探讨以下问题：
- 为什么batchsize小的时候，模型训练很不稳定
- 为什么预测超出训练范围的点时，模型会出现很大的误差
- 如果要增加模型能够学习的范围，应该怎么改进
## 实验三（Optional，for those who have access to gpus）
本地部署一个文生图模型来玩玩！（让我们假设你的电脑有显卡，多小无所谓，总能找到合适的）
推荐的路径：
- 本地安装Nvidia相关工具
- nvidia-smi查看显卡信息
- 根据显卡信息，去huggingface上找到**合适大小**的模型
- 下载模型
- 运行模型，生成图片

这个过程中，你需要学会：
    - 根据显卡大小，估算可以最大运行的模型（calculating）
    - 如何下载huggingface上的模型（downloading）
    - 如何使用huggingface的模型生成图片（infering）

实际上，这些技能都是非常有用的。我们强烈推荐您在**OpenAI/Claude**等工具的帮助下，完成这一小节的任务！
每个环节你都可能会遇到问题，没关系，使用**OpenAI/Claude**等工具来帮助你解决问题。
可能但不限于：
- huggingface下载速度慢
- 模型太大，显存不够
- huggingface连不上/无权限
...
# 更多
实践出真知！