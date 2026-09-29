# mini_gpt

一个用 PyTorch 从头实现的最小自回归语言模型项目，沿 Stanford CS336 Assignment 1 的顺序学习字节级 BPE、Transformer、训练与文本生成。本仓库的模型、优化器和分词器实现放在根目录；`test/` 是本项目的测试，`config/` 是运行配置。使用项目只需阅读本 README。

模型由 token Embedding、带 RoPE 的因果多头自注意力、RMSNorm、SwiGLU、Transformer Block 和输出投影组成。训练使用交叉熵与自实现的 AdamW，支持梯度裁剪、余弦学习率调度、周期 checkpoint 和断点续训；推理支持温度和 top-p 采样。当前已验证小样本分词、端到端训练、周期保存、中断续训、checkpoint 加载和文本生成。配置是流程起点，**并非已训练好的模型或保证收敛的超参数**。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `BPE.py`、`PrepareTraining.py`、`TokenizeData.py` | BPE 实现、加载 GPT-2 词表、把 UTF-8 文本编码为 token 文件 |
| `Embedding.py`、`Attention.py`、`RoPE.py`、`RMSNorm.py`、`SwiGLU.py`、`TransformerBlock.py`、`TransformerLM.py` | 模型组件及完整语言模型 |
| `CrossEntropy.py`、`AdamW.py`、`Data.py`、`Train.py`、`TrainLoop.py`、`TrainingScript.py`、`Checkpoint.py` | 损失、优化器、取批次、训练循环及保存/加载 |
| `GradientClipping.py`、`LearningRateSchedule.py` | 梯度裁剪与余弦学习率调度，由训练循环按配置调用 |
| `DownloadCorpora.py`、`RunTraining.py`、`Generate.py`、`RunGeneration.py` | 下载语料、训练入口和生成入口 |
| `assets/gpt2/`、`assets/smoke_stories.txt` | 项目自带的 GPT-2 词表/merges 与冒烟测试小样本 |
| `config/`、`test/` | 示例配置、测试 |
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

GPT-2 词表和合并规则已随项目放在 `assets/gpt2/`，冒烟样本放在 `assets/smoke_stories.txt`。安装依赖后即可使用项目提供的数据准备、训练与推理入口；无需额外下载分词器文件。资源来源及许可说明见 `assets/gpt2/NOTICE.md`。

## 五分钟流程验证

以下命令在项目根目录执行。`--smoke` 从项目内置的英文小故事生成互不重叠的训练/验证文本，再编码成 `uint16` token 文件：

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

### 可选训练选项

以下字段可直接写入配置 JSON，默认值即当前行为（关闭）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `checkpoint_interval` | 不填 | 每隔多少步把 checkpoint 原子替换写入 `checkpoint_path`。**同一个文件被覆盖**，不会累积出多份权重 |
| `max_grad_norm` | 不填 | 梯度裁剪的 L2 范数上限；不填则不裁剪 |
| `warmup_iters` | `0` | 线性 warmup 步数；**必须与 `cosine_cycle_iters` 同时填写**才生效 |
| `cosine_cycle_iters` | 不填 | 余弦衰减的周期步数，一般设成 `max_steps`。不填则整个调度关闭 |
| `min_learning_rate` | `0.0` | 衰减到的最小学习率 |
| `resume_from` | 不填 | 从该 checkpoint 继续训练。相对路径同样基于项目根目录解析 |

中断后恢复训练的做法是：确认 `checkpoint_path` 里的权重仍在，把同一个配置原样重跑即可——训练会从 checkpoint 记录的步号继续，并把日志中超出该步号的记录裁掉。若要**换个步数继续**（例如把 `max_steps` 调大），把 `resume_from` 指向原 `checkpoint_path` 即可；`checkpoint_path` 应当仍是同一路径，否则续训中途的结果不会落到你预期的位置。

长训练建议同时设置 `checkpoint_interval` 和余弦调度。前者让中断可恢复，后者让后期学习率自然衰减、不再震荡。

## 从 checkpoint 生成文本

配置中的模型结构必须与 checkpoint 一致，分词器也必须与训练数据使用的词表一致。下面的命令默认加载 `config/tinystories.json` 指向的 `runs/tinystories/final.pt`：

```powershell
python RunGeneration.py --config config/tinystories.json --prompt "Once upon a time" --max-new-tokens 120 --temperature 0.8 --top-p 0.9 --device cpu
```

可以用 `--checkpoint` 指定其他权重；GPU 可改用 `--device cuda`。程序打印 checkpoint 的训练步数和完整生成文本。`--max-new-tokens` 是最多新增的 token 数，遇到结束 token 可能提前停止；每步最多只把配置中 `context_length` 个最近 token 送入模型，因此生成结果可以比上下文窗口长，但模型不会继续看到更早的内容。

## 测试与当前限制

```powershell
python -m pytest test -q
```

`test/test_training_reliability.py` 覆盖训练可靠性：绝对步号的续训、训练中断后指标与 checkpoint 的存活、原子写入，以及日志中超前于 checkpoint 的记录在续训时被裁剪。它已纳入常规回归命令。测试只使用项目内的源码和资源。

- 指标 JSONL 按评估点追加写入，checkpoint 按 `checkpoint_interval` 原子替换同一个文件；两者在**下一个保存点之前**中断，仍会丢失该区间内未保存的进度。周期保存不等于实时保存。
- 梯度裁剪与余弦学习率调度**需在配置中显式开启**：`max_grad_norm` 不填则不裁剪；`warmup_iters` 必须与 `cosine_cycle_iters` 同时填写，只填前者会被静默忽略（不生效也不报错）。
- 生成入口只加载本项目 `TransformerLM` 的 checkpoint，并非通用的 GPT-2 或其他市售模型权重加载器。
- 中文可以被字节级 tokenizer 编码，但当前训练/推理入口固定使用 GPT-2 词表；改用新词表时，需要保持预处理、模型配置和推理一致，不能直接复用旧 checkpoint。
