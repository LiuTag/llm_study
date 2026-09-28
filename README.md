# mini_gpt

一个用 PyTorch 从头实现的最小自回归语言模型项目，沿 [Stanford CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics) 的顺序学习字节级 BPE、Transformer、训练与文本生成。本仓库的模型、优化器和分词器实现放在根目录；`test/` 是本项目的测试，`config/` 是运行配置，`doc/` 保存原理笔记，**使用项目只需阅读本 README**。

模型由 token Embedding、带 RoPE 的因果多头自注意力、RMSNorm、SwiGLU、Transformer Block 和输出投影组成。训练使用交叉熵与自实现的 AdamW；推理支持温度和 top-p 采样。当前已验证小样本分词、两步端到端训练、checkpoint 加载和文本生成。配置是流程起点，**并非已训练好的模型或保证收敛的超参数**。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `BPE.py`、`PrepareTraining.py`、`TokenizeData.py` | BPE 实现、加载 GPT-2 词表、把 UTF-8 文本编码为 token 文件 |
| `Embedding.py`、`Attention.py`、`RoPE.py`、`RMSNorm.py`、`SwiGLU.py`、`TransformerBlock.py`、`TransformerLM.py` | 模型组件及完整语言模型 |
| `CrossEntropy.py`、`AdamW.py`、`Data.py`、`Train.py`、`TrainLoop.py`、`TrainingScript.py`、`Checkpoint.py` | 损失、优化器、取批次、训练循环及保存/加载 |
| `GradientClipping.py`、`LearningRateSchedule.py` | 已单独实现；目前尚未接入实际训练循环 |
| `DownloadCorpora.py`、`RunTraining.py`、`Generate.py`、`RunGeneration.py` | 下载语料、训练入口和生成入口 |
| `config/`、`test/`、`doc/` | 示例配置、测试、可选的原理笔记 |
| `data/`、`runs/` | 本地语料及 token 文件、训练指标及 checkpoint；已被 Git 忽略 |

## 环境准备

建议使用 Python 3.12。从项目根目录运行以下命令；Windows PowerShell 中可以使用 `py -3.12` 代替 `python`：

```powershell
python -m venv .venv
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
# Linux/macOS：source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` 包含 `torch`。如果要使用 NVIDIA GPU，建议在安装其余依赖前，先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 为本机选择合适的 CUDA 版；仅安装 CUDA Toolkit 不代表当前 PyTorch 能使用 GPU。CPU 可以用于测试和小规模冒烟训练。

**当前资源依赖：**`PrepareTraining.py` 和 `RunGeneration.py` 仍从 `assignment1-basics/tests/fixtures/` 读取 GPT-2 的 `gpt2_vocab.json` 与 `gpt2_merges.txt`；冒烟数据准备还会读取该目录下的 TinyStories 小样本。若项目根目录没有 `assignment1-basics/`，先运行：

```powershell
git clone --depth 1 https://github.com/stanford-cs336/assignment1-basics.git assignment1-basics
```

这是当前版本的外部资源依赖，不需要把整个官方仓库复制进本项目的 GitHub 仓库。尚未将这两份词表文件整理为项目自带资源，因此**只克隆本仓库并安装依赖，还不能直接进行数据准备或推理**。

## 五分钟流程验证

以下命令在项目根目录执行。`--smoke` 从官方仓库附带的小样本生成互不重叠的训练/验证文本，再编码成 `uint16` token 文件：

```powershell
python PrepareTraining.py --smoke --output-dir data/smoke
python RunTraining.py --config config/smoke.json --device cpu
python RunGeneration.py --config config/smoke.json --checkpoint runs/smoke/final.pt --device cpu --prompt "Once upon a time" --max-new-tokens 50
```

训练结束后可在 `runs/smoke/` 看到 `metrics.jsonl` 和 `final.pt`。冒烟配置只更新两步，目的是检查数据、前向/反向、优化器、保存及加载是否连通；生成文本通常不会流畅。若有可用 CUDA，也可以去掉训练与推理命令中的 `--device cpu`。

## 训练 TinyStories / OpenWebText

当前预处理入口使用 GPT-2 词表，并调用本项目的 `BPETokenizer` 编码。输入是 UTF-8 文本；输出的 `train.bin`、`val.bin` 是**无文件头、小端 `uint16` token ID 序列**。训练集和验证集必须使用同一词表；`metadata.json` 记录 token 数。语料、分词结果和 checkpoint 均放在被 Git 忽略的 `data/`、`runs/` 中。

TinyStories：

```powershell
python DownloadCorpora.py tinystories
python PrepareTraining.py --train-text data/raw/tinystories/TinyStoriesV2-GPT4-train.txt --val-text data/raw/tinystories/TinyStoriesV2-GPT4-valid.txt --output-dir data/tinystories
python RunTraining.py --config config/tinystories.json --probe-steps 100
python RunTraining.py --config config/tinystories.json
```

课程使用的 OpenWebText **子集**（不是完整 OpenWebText）：

```powershell
python DownloadCorpora.py owt --extract-owt
python PrepareTraining.py --train-text data/raw/owt/owt_train.txt --val-text data/raw/owt/owt_valid.txt --output-dir data/owt
python RunTraining.py --config config/owt.json
```

完整语料下载与编码可能耗时较长，需要预留磁盘空间。下载脚本使用 `.part` 保存未完成文件并尝试续传。`--probe-steps 100` 将输出写到独立的 `*_probe` 文件，不覆盖正式训练结果。`config/tinystories.json` 和 `config/owt.json` 当前均为 256-token 上下文、1000 步的起始配置；1000 步不代表已经充分训练。`config/smoke.json` 的上下文为 64 token。

## 从 checkpoint 生成文本

配置中的模型结构必须与 checkpoint 一致，分词器也必须与训练数据使用的词表一致。下面的命令默认加载 `config/tinystories.json` 指向的 `runs/tinystories/final.pt`：

```powershell
python RunGeneration.py --config config/tinystories.json --prompt "Once upon a time" --max-new-tokens 120 --temperature 0.8 --top-p 0.9 --device cpu
```

可以用 `--checkpoint` 指定其他权重；GPU 可改用 `--device cuda`。程序打印 checkpoint 的训练步数和完整生成文本。`--max-new-tokens` 是最多新增的 token 数，遇到结束 token 可能提前停止；每步最多只把配置中 `context_length` 个最近 token 送入模型，因此生成结果可以比上下文窗口长，但模型不会继续看到更早的内容。

## 测试与当前限制

```powershell
python -m pytest test -q --ignore=test/test_training_reliability.py
```

部分本地 adapter 测试以及数据准备/推理测试依赖上面克隆的官方仓库资源。`test/test_training_reliability.py` 是下一阶段的验收测试，现阶段预期失败，因此在当前回归命令中暂时排除；不要把这个排除项理解为功能已通过。

- 当前 `TrainingScript.py` **训练结束后**才写入指标 JSONL 和最终 checkpoint；中途中断会丢失本次未保存的进度，周期保存、实时日志和断点续训尚未完成。
- 梯度裁剪和余弦学习率函数已实现，但训练目前使用固定学习率，尚未调用这两个函数。
- 生成入口只加载本项目 `TransformerLM` 的 checkpoint，并非通用的 GPT-2 或其他市售模型权重加载器。
- 中文可以被字节级 tokenizer 编码，但当前训练/推理入口固定使用 GPT-2 词表；改用新词表时，需要保持预处理、模型配置和推理一致，不能直接复用旧 checkpoint。
