# SAST2025年暑培 - 神经网络与 PyTorch 入门资料仓库

## 课前准备

以下是一些必要的课前准备工作。

---

### 1. **Conda 环境准备**  
如果你之前已经在 Python 课程中配置过 Conda 环境，直接创建一个新环境并跳过这一节即可。没有接触过的同学也不用担心，按照以下步骤来配置你的开发环境，确保你能轻松运行课程所需的代码。

#### **什么是 Conda？**  
**Conda** 是一个跨平台的开源软件包管理系统和环境管理系统，它能够让我们在不同项目中创建隔离的运行环境，避免不同库版本冲突的问题。在 Windows、macOS 和 Linux 上都可以使用 Conda。通过 Conda，你可以快速安装、管理和切换不同的环境。

**Anaconda** 是一个包含 Conda 的预配置工具集，内置了常见的 Python 库，推荐大家使用 **Anaconda**，因为它已经为数据科学和机器学习等领域做了很多优化。

#### **安装 Conda 和 Anaconda**
1. **下载 Anaconda**  
   推荐通过 **TUNA 镜像站** 下载 Anaconda 安装包，这样可以大大提高下载速度：
   [TUNA Anaconda 镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

2. **安装 Anaconda**  
   按照官网的步骤进行安装，并根据操作系统选择相应版本。

3. **创建 Conda 环境**  
   打开命令行终端，使用以下命令创建一个新的环境：
   ```bash
   conda create -n myenv python=3.8
   ```

    你可以将 `myenv` 替换为你喜欢的环境名称。

4. **激活环境**
   创建好环境后，激活它：

   ```bash
   conda activate myenv
   ```

5. **安装依赖库**

在conda环境下安装依赖库时，我们可以使用 **Conda** 和 **Pip**。一般来说，**Conda** 用于安装复杂依赖和大型框架（如 PyTorch），它能自动解决依赖问题；而 **Pip** 适合安装轻量级库或 Conda 中没有的库。一般人们直接使用**Pip**就完事了，如果不涉及复杂依赖。
值得注意的是，通过**Pip**和**Conda**安装的包在环境中都是可以正常使用的，但是它们之间的依赖问题不会被同时考虑。安装时，**Conda**考虑通过它安装的依赖问题，**Pip**则考虑通过它安装的依赖问题。尽量只使用一种。

比如conda来安装一个pytorch（举例）
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

而 Huggingface 的 `transformers` 库则需要通过 **Pip** 安装：

```bash
pip install transformers
```

一般你不用自己安装指定版本的具体包，因为人家一般在根目录下写好了requirements.txt文件，你只需要运行以下命令即可(本仓库也一样)：

```bash
pip install -r requirements.txt
```

### 2. **Wandb 注册（Optional）**

**Wandb**（Weights and Biases）是一个用于机器学习项目的工具，能够帮助你追踪实验过程中的各项指标、超参数调整和模型可视化。通过 Wandb，你可以更轻松地管理和可视化训练过程。

#### **注册与使用**：

1. **注册 Wandb 账号**
   访问 [Wandb 官网](https://wandb.ai/home) 注册账号，完成基础设置。

2. **登录 Wandb**
   进入环境，然后安装python包
   ```
   conda activate myenv
   pip install wandb
   ```
   完成注册后，使用以下命令在本地命令行登录：

   ```bash
   wandb login
   ```
   随后将官网得到的token输入即可

3. **国内替代方案 - Swanlab**
   在国内使用 Wandb 可能会遇到连接问题，因此可以考虑使用 **Swanlab**，它是 Wandb 在国内的优化版本。访问 [Swanlab 官网](https://swanlab.cn/) 进行注册。逻辑与wandb类似

4. **集成 Wandb 到 PyTorch 项目**
   在项目中，使用以下代码集成 Wandb 进行实验追踪：

   ```python
   import wandb
   # initialize Wandb
   wandb.init(project="my-project-name")

   # Upload some metric
   wandb.log({"accuracy": 0.95, "loss": 0.05},step=1)
   ```

### 3. **VSCode Jupyter Notebook 语言扩展**
1. 打开 **VSCode**。
2. 按下 **Ctrl+Shift+X** 打开扩展商店。
3. 搜索并安装 **Jupyter** 扩展（官方扩展）。这会使 VSCode 支持 `.ipynb` 格式的文件，并能够运行和调试 Jupyter Notebook。

安装后，你就可以在 VSCode 中直接运行 Jupyter Notebooks，进行交互式编程了！

### 4. **PyTorch 安装**

**PyTorch** 是目前最流行的深度学习框架之一，它被广泛应用于计算机视觉、自然语言处理和生成模型等领域。

#### **安装步骤**：
访问官网：
https://pytorch.org/get-started/locally/
选择自己的实际配置，粘贴官网生成的命令到终端中执行即可。
比如说我的配置是：windows+pip+cuda 11.8
```bash
pip3 install torch torchvision torchaudio -i https://download.pytorch.org/whl/cu118
```
如果觉得安装的太慢了，可以考虑使用清华源：**（记住这个技巧）**
```bash
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装完成后，通过以下 Python 代码验证是否成功：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 检查是否能使用 GPU
```

### 5. **Huggingface 注册（Optional）**

**Huggingface** 是一个顶尖的人工智能平台，拥有大量的预训练模型、数据集和代码。你可以利用 Huggingface 提供的预训练模型进行快速原型设计，并通过其平台进行模型训练和分享。
甚至还可以在上面看daily paper，就像早上起来读报的老大爷。
#### **注册与使用**：

1. **注册 Huggingface 账号**
   访问 [Huggingface 官网](https://huggingface.co/) 注册账号。

2. **安装 Huggingface 的 `transformers` 库**
   使用以下命令安装 Huggingface 的 `transformers` 库，来加载和使用预训练模型：

   ```bash
   pip install transformers
   ```

3. **加载预训练模型**
   安装后，你可以快速加载预训练的 BERT 模型，进行文本分类、生成任务等：

   ```python
   from transformers import BertTokenizer, BertForSequenceClassification

   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   ```

注：huggingface如果访问困难，或者需要下载大文件，可以使用hf-mirror.com国内镜像站平替
此时，需要引入环境变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 6. **其他的网站**
- **d2l.ai**
    这是一本非常好的深度学习教材，提供了大量的代码示例和练习题。可以通过以下链接访问：https://d2l.ai/
    有中文版可以切换，且附有仓库
    ```
    git clone https://github.com/d2l-ai/d2l-en.git
    ```
- **谷歌开发者文档**
   https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises
- **Tensorflow的play ground**
    https://playground.tensorflow.org/
    可视化很好，比较好玩
- **给小孩的机器学习教程**（字面意思）
    https://machinelearningforkids.co.uk/