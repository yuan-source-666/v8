# v8 —— 本地自训小 LLM（Hybrid Mamba-Transformer + 自进化驱动闭环）

> 路线 C：预训练 → SFT → DPO → 自我进化闭环
> 目标硬件：Intel Core Ultra 7 255H / 32GB 内存 / CPU-only（无独立显卡）
> 架构依据：《LLM自进化架构设计.md》v1.2（见 `docs/`）

本仓库提供一个**真实可运行**的小语言模型自训练脚手架：从零搭建
「主流 Transformer 底座 + 前沿 Hybrid Mamba-2 混合 + 自我进化闭环 + 驱动信号层
（恐惧/欲望的 L2 稳态驱动机制）」。所有代码均为真实实现，禁止 TODO 占位。

## 1. 目录结构

```
v8/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── config/
│   └── mini.yaml                 # Mini ~20M 与 Tiny ~54M 两档配置
├── docs/
│   └── LLM自进化架构设计.md        # 架构蓝皮书拷贝（v1.2）
├── data/
│   ├── prepare.py                # 原始文本 → tokenized 缓存管线（一次性落盘）
│   ├── sample_notes.txt          # 语料放置说明
│   └── corpus/                   # （用户自建）原始语料目录：*.txt，UTF-8
├── model/
│   ├── __init__.py
│   ├── mamba2.py                 # Mamba-2 SSD 层（纯 PyTorch）
│   ├── attention.py              # GQA + RoPE 自注意力
│   ├── hybrid.py                 # 混合骨干层（Mamba-2 与 Attention 按 2:1 交替 + SwiGLU）
│   └── model.py                  # CausalLM：embedding → 主骨干 → lm_head（可权重绑定）
├── drives/
│   ├── __init__.py
│   ├── state.py                  # 持续内部状态 S(t)
│   ├── signal.py                 # 稳态偏差信号 D=|S−setpoint|（欲望/恐惧）
│   └── rewards.py                # 把 D 接入 intrinsic reward / loss 正则项
├── train/
│   ├── __init__.py
│   ├── train.py                  # 预训练主循环
│   ├── sft.py                    # 最小可运行 SFT（内置示例指令集）
│   └── dpo.py                    # 最小可运行 DPO（内置枚举偏好对）
├── infer/
│   ├── __init__.py
│   └── sample.py                 # 采样生成 + self-consistency 多数投票
└── scripts/
    ├── run_mini.ps1              # Windows 一键启动（prepare → train）
    └── run_mini.sh               # Linux/macOS 一键启动（prepare → train）
```

## 2. 安装依赖

```bash
# 建议 Python 3.10–3.11
pip install -r requirements.txt
```

注意：本机为 CPU-only，请安装 **CPU 版 PyTorch**，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`intel-extension-for-pytorch`（IPEX）为可选加速项：安装失败时训练脚本会自动退回
原生 PyTorch + oneDNN（不影响正确性，仅影响吞吐）。

## 3. 快速开始（Windows）

```powershell
cd E:\agi研发项目\v8
powershell -ExecutionPolicy Bypass -File scripts\run_mini.ps1
```

Linux / macOS：

```bash
cd /path/to/v8
bash scripts/run_mini.sh
```

一键脚本会自动完成：**数据准备（prepare）→ 预训练（train）**。

手动分步：

```bash
# 1) 准备语料：把原始 .txt 文本放入 data/corpus/（UTF-8），然后：
python data/prepare.py --config config/mini.yaml

# 2) 若暂无语料，可用内置演示语料跑通整条管线（仅用于验证链路）：
python data/prepare.py --config config/mini.yaml --demo

# 3) 预训练（Mini 档）
python train/train.py --config config/mini.yaml --profile mini

# 4) 采样生成
python infer/sample.py --config config/mini.yaml --profile mini --ckpt out/mini/best.pt --prompt "从前有一个"

# 5) SFT（加载预训练权重继续微调）
python train/sft.py --config config/mini.yaml --profile mini --init out/mini/best.pt

# 6) DPO（加载 SFT 权重做偏好对齐）
python train/dpo.py --config config/mini.yaml --profile mini --init out/mini/sft.pt
```

## 4. 冒烟验证（验收标准 1）

在项目根目录执行（Mini 档，前向 + 反向各一步不报错）：

```bash
python -c "from model.model import build_model; import torch; m = build_model('config/mini.yaml', 'mini'); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); x = torch.randint(0, m.vocab_size, (2, 128)); logits, _ = m(x); loss = logits.mean(); loss.backward(); print('SMOKE_OK', tuple(logits.shape), 'params(M)=', round(sum(p.numel() for p in m.parameters())/1e6, 3))"
```
说明：`m(x)` 返回 `(logits, loss)` 元组，需解包取 `logits` 后再做 `.mean()`；随机输入上限用 `m.vocab_size`（运行时以 tokenizer 真实词表为准，gpt2 为 50257）。

## 5. 训练流程与自进化闭环

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 预训练 | `train/train.py` | CPU + IPEX/oneDNN + bf16 混合精度 + 梯度累积 + warmup/cosine + checkpoint 续训；可选开启 drives 内在奖励 |
| SFT | `train/sft.py` | 内置示例指令集，监督微调 |
| DPO | `train/dpo.py` | 内置枚举偏好对，ref 模型冻结 |
| 推理细化 | `infer/sample.py` | 温度/top-k/top-p 采样 + self-consistency 多数投票 |
| 驱动信号层 | `drives/` | L2 稳态机制：内部状态 S(t) → 偏差信号 D → intrinsic reward / loss 正则 |

`train/train.py --use-drives` 会把驱动信号层的正则项接入训练损失（drives 三件套真实参与训练，非摆设）。

## 6. 两档配置

| 档位 | 参数量(约) | d_model | 层数 | 布局 | 定位 |
|------|-----------|---------|------|------|------|
| `mini` | ~20M | 256 | 6 | [M,M,A]×2 | 纯机制验证，日级产出，冒烟验证用 |
| `tiny` | ~54M | 512 | 6 | [M,M,A]×2 | 本机主战场，真实闭环 |

> 注：上表参数量为 **gpt2 词表（50257）** 下的实测值；tied embedding（`50257×d_model`）占了大头（mini ≈12.9M、tiny ≈25.7M）。若需贴近原始 ~10M/~33M 设计，可改用小词表 tokenizer 或显式固定 `vocab_size`。
详细超参见 `config/mini.yaml`。

## 7. 常见问题

- **没有语料？** 见 `data/sample_notes.txt`；临时验证链路用 `python data/prepare.py --demo`。
- **IPEX 装不上？** 不影响运行，脚本自动跳过 IPEX，仅提示。
- **续训**：`python train/train.py --config ... --resume out/mini/best.pt`。
- **训练速度慢？** 确保 `torch.set_num_threads(16)` 生效；在 BIOS/电源选项中开启高性能。
- **model.py 直接跑报 ImportError？** 用模块方式：`python -m model.model --config config/mini.yaml --profile mini`（相对导入要求以包运行）。
- **核数不足 16？** `torch.set_num_threads(16)` 只影响吞吐、不影响正确性；实际核数少于 16 时把数值改成真实核数即可。

（内容由 AI 生成，仅供参考）
