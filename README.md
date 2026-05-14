# Large Language Model Agents Are Not Always Faithful Self-Evolvers

> Official code repository for the ICML 2026 paper:
> **Large Language Model Agents Are Not Always Faithful Self-Evolvers**
>
> *Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yang Deng, Yanyan Zhao, Wanxiang Che, Bing Qin, Ting Liu*

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## 📖 Overview

Self-evolving LLM agents continually improve by accumulating and reusing past experience. But do they truly **rely** on that experience? This work presents the first systematic study of **experience faithfulness** — the causal dependence of an agent's behavior on the experience it is given.

We evaluate four representative self-evolving frameworks across **10 LLM backbones** and **9 environments** under controlled causal interventions on both **raw experience** (full trajectories) and **condensed experience** (distilled summaries / heuristics). Our results expose a striking asymmetry: agents reliably exploit raw experience, but largely ignore or misuse condensed experience — even when it is the only experience provided.

This repository provides the code to reproduce all four faithfulness experiments and the IG (Information Gain) score analysis used in the paper.

## 🗂️ Repository Layout

```
.
├── ExpeL/                # Offline single-agent (ExpeL) — ALFWorld / HotpotQA / WebShop
├── dynamic-cheetsheet/   # Online single-agent (Dynamic Cheatsheet) — AIME / GameOf24 / GPQA-Diamond / MMLU-Pro Eng.
├── ReasoningBank/        # Online single-agent (ReasoningBank) — WebArena
└── cdt/                  # IG-Score analysis
```

## 🧪 Intervention Types

All four code bases share a unified set of intervention flags. They correspond exactly to the interventions defined in §3 of the paper.

| Flag | Component | Option | Meaning |
|---|---|---|---|
| `faithfulness_experiment` | switch | `true` / `false` | Master switch. When `false`, runs the unperturbed baseline. |
| `fewshot_modification_type` | **Raw** experience (retrieved trajectories / few-shots) | `empty` | Strip all semantic content but keep the formatting cues (cf. paper "Empty"). |
|  |  | `shuffle` | Randomly shuffle the order of steps within each trajectory. |
|  |  | `irrelevant` | Replace with trajectories sampled from unrelated tasks. |
|  |  | `none` / `None` | No modification. |
| `insights_modification_type` | **Condensed** experience (insights / cheatsheet / memory bank) | `empty` | Remove semantic content while keeping the surrounding template. |
|  |  | `corrupt` / `corrupted` | Randomly distort key elements (action references, etc.) to break internal coherence. |
|  |  | `irrelevant` | Replace with a generic / task-agnostic summary. |
|  |  | `filler_tokens` | Replace content with length-matched placeholder tokens (e.g. `%$#@&`). |
|  |  | `wo` | Remove the condensed-experience block entirely (full ablation, `w/o cond`). |
|  |  | `none` / `None` | No modification. |

## 🛠️ Common Prerequisites

- Python ≥ 3.10 (recommend 3.10/3.11)
- [vLLM](https://github.com/vllm-project/vllm) for local serving of open-weight models
- An OpenAI-compatible API endpoint for closed-source backbones 

Before running anything, configure your model credentials. Most entry points read from environment variables:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-openai-compatible-endpoint/v1"
```

Each sub-project has its own `requirements.txt` / environment requirements — please follow the upstream repos linked below for complete environment setup, then install our additional patches.

---

## 1. ExpeL — Offline Single-Agent

> Reference: [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)

### Setup

1. Follow the official ExpeL repository to set up the conda env and download the **ALFWorld**, **HotpotQA** and **WebShop** datasets.
2. For WebShop, also start the local WebShop server according to its official instructions.
3. Configure your OpenAI-compatible `base_url` and `api_key`.
4. Install ExpeL deps:

```bash
cd ExpeL
pip install -r requirements.txt
```

### Pipeline

The full pipeline consists of **three stages** (already chained inside each script):

1. **Experience Gathering** — `train.py` runs the agent on training tasks and stores raw trajectories.
2. **Insights Extraction** — `insight_extraction.py` distills condensed insights from those trajectories.
3. **Faithfulness Evaluation** — `eval.py` evaluates the agent on test tasks under the unperturbed baseline and under each intervention.

```bash
cd ExpeL/scripts
bash run_alfworld.sh   # ALFWorld (embodied)
bash run_hotpot.sh     # HotpotQA  (knowledge QA)
bash run_webshop.sh    # WebShop   (web interaction)
```

Each script enumerates all 7 interventions: `fewshot ∈ {empty, shuffle, irrelevant}` and `insights ∈ {empty, corrupted, irrelevant, filler_tokens}`. Logs and checkpoints are written to `ExpeL/logs/`.

### Manual Invocation

```bash
python eval.py benchmark=alfworld \
  load_run_name=extracted_insights/insights-extraction-alfworld_14B \
  run_name=my_eval_run \
  agent.llm=Qwen/Qwen3-14B \
  agent.fewshot_strategy=task_similarity \
  agent.faithfulness_experiment=true \
  agent.fewshot_modification_type=shuffle \
  agent.insights_modification_type=none
```

---

## 2. Dynamic Cheatsheet — Online Single-Agent

> Reference: [suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet)

### Setup

1. Follow the official Dynamic Cheatsheet repo for environment setup.
2. Configure your OpenAI / Gemini credentials.
3. Datasets (AIME 2024/2025, GameOf24, GPQA-Diamond, MMLU-Pro Engineering) ship with the upstream repo under `dynamic-cheetsheet/data/`.

### Run

```bash
cd dynamic-cheetsheet/scripts
bash run.sh         # full sweep over all 5 tasks × all interventions
```

`scripts/run.sh` iterates over `tasks=("AIME_2024" "AIME_2025" "GameOf24" "GPQA_Diamond" "MMLU_Pro_Engineering")` and for each task runs the unperturbed baseline plus six interventions (`insights ∈ {wo, corrupted, irrelevant, filler_tokens}` and `fewshot ∈ {shuffle, irrelevant}`).

### Manual Invocation

```bash
python run_benchmark.py \
  --task GPQA_Diamond \
  --approach DynamicCheatsheet_RetrievalSynthesis \
  --model_name openai/gpt-4o \
  --save_directory TEST_RESULTS \
  --max_n_samples 100 \
  --faithfulness_experiment \
  --insights_modification_type corrupted \
  --fewshot_modification_type None
```

---

## 3. ReasoningBank — Online Single-Agent (Condensed-Only)

> Reference paper: ReasoningBank (Ouyang et al., 2025). Built on top of [WebArena](https://github.com/web-arena-x/webarena) + [BrowserGym](https://github.com/ServiceNow/BrowserGym).

### Setup

1. **Set up WebArena Docker images** — follow the [WebArena official guide](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md) to build / pull the five site images:
   
   - `shopping_final_0712` 
   - `shopping_admin_final_0719`
   - `postmill-populated-exposed-withimg` 
   - `gitlab-populated-final-port8023` 
   - OpenStreetMap tile server — or use the public `https://www.openstreetmap.org`
2. Install Python deps and Playwright:
   ```bash
   cd ReasoningBank
   pip install -r requirements.txt
   playwright install chromium
   ```
3. Configure backbone credentials.

### Run

```bash
cd ReasoningBank/scripts
bash run_shopping.sh    # WebArena Shopping
bash run_cms.sh         # WebArena Shopping-Admin (CMS)
bash run_reddit.sh      # WebArena Reddit
bash run_gitlab.sh      # WebArena GitLab  
bash run_map.sh         # WebArena Map   
```

Each script runs:
1. Memory-bank build-up (`--use_bank True`)
2. Bank-disabled baseline (`--use_bank False`)
3. Four insights interventions (`wo`, `corrupt`, `irrelevant`, `filler_tokens`)

### Manual Invocation

```bash
python pipeline.py \
  --website shopping \
  --model_name Qwen/Qwen3-32B \
  --faithfulness_experiment True \
  --insights_modification_type corrupt
```

Available arguments:
- `--website ∈ {shopping, shopping_admin, gitlab, reddit, map}`
- `--use_bank ∈ {True, False}`
- `--faithfulness_experiment True --insights_modification_type ∈ {wo, corrupt, irrelevant, filler_tokens}`
- `--faithfulness_experiment True --fewshot_modification_type ∈ {shuffle, irrelevant}` 

---

## 4. IG-Score — Information Gain Analysis

> Built on the [Context Denoising Training](https://github.com/LCM-Lab/context-denoising-training) codebase. 

### Setup

1. Follow the official Context Denoising Training repo to install dependencies (`transformers`, `accelerate`, `flash-attn`, `faiss`, …).
2. Place the formatted prompt JSON files under `cdt/src/inputs/`. The expected files (referenced by the scripts) are:
   - `input_prompt_with.json` — prompts with original condensed experience
   - `input_prompt_wo.json` — without condensed experience
   - `input_prompt_corrupt.json`
   - `input_prompt_irrelevant.json`
   - `input_prompt_empty.json`
   - `input_prompt_filler_tokens.json`

   These can be exported from the ExpeL / ReasoningBank / Dynamic Cheatsheet runs above (raw prompt traces).

### Run

The repository provides per-scale launcher scripts:

```bash
cd cdt/src/scripts
bash run_8B.sh
bash run_14B.sh
bash run_32B.sh
```

Each script sets `CUDA_VISIBLE_DEVICES` and iterates `pipeline_rb.py` over all input variants:

```bash
python pipeline_rb.py \
  --full_data_path ./inputs/input_prompt_with.json \
  --only_ig True \
  --model_path Qwen/Qwen3-32B
```

Results (per-token / per-layer IG scores) are written to `cdt/results/`. Use `stats_igscore.py` and `stats_frscore.py` to aggregate.

---

## 📑 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhao2026faithful,
  title     = {Large Language Model Agents Are Not Always Faithful Self-Evolvers},
  author    = {Zhao, Weixiang and Wang, Yingshuo and Zhang, Yichen and Deng, Yang and
               Zhao, Yanyan and Che, Wanxiang and Qin, Bing and Liu, Ting},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 🙏 Acknowledgements

Our code builds upon the following excellent open-source projects:

- [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)
- [suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet)
- [WebArena](https://github.com/web-arena-x/webarena) & [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [Context Denoising Training](https://github.com/LCM-Lab/context-denoising-training) 

---
---

<a id="中文"></a>

## 📖 项目简介

自演化（Self-Evolving）大模型 Agent 通过不断累积并复用过往经验来持续改进。但它们是否**真的依赖**这些经验？本工作首次系统性地研究了 **经验忠实性（Experience Faithfulness）**——即 Agent 行为对其所获得经验的因果依赖程度。

我们在 **10 个 LLM backbone × 9 个环境** 上对 4 个代表性自演化框架进行了受控因果干预实验，分别针对**原始经验（raw experience，完整轨迹）** 和 **凝练经验（condensed experience，蒸馏后的摘要 / 启发式）**。结果揭示了一个显著的不对称现象：Agent 始终依赖原始经验，却普遍忽视甚至误用凝练经验——即便凝练经验是它唯一可用的经验。

本仓库提供了论文中四个忠实性实验，以及 IG（Information Gain）分析的完整复现代码。

## 🗂️ 目录结构

```
.
├── ExpeL/                # 离线单 Agent（ExpeL）—— ALFWorld / HotpotQA / WebShop
├── dynamic-cheetsheet/   # 在线单 Agent（Dynamic Cheatsheet）—— AIME / GameOf24 / GPQA-Diamond / MMLU-Pro Eng.
├── ReasoningBank/        # 在线单 Agent（ReasoningBank）—— WebArena
└── cdt/                  # IG-Score 分析
```

## 🧪 干预类型说明

四份代码使用统一的干预参数命名，与论文 §3 中定义的干预类型一一对应。

| 参数 | 作用对象 | 取值 | 含义 |
|---|---|---|---|
| `faithfulness_experiment` | 总开关 | `true` / `false` | 设为 `false` 时跑未干预的 baseline。 |
| `fewshot_modification_type` | **原始经验**（检索到的轨迹 / few-shot） | `empty` | 清空语义内容，仅保留模板提示语（论文 "Empty"）。 |
|  |  | `shuffle` | 随机打乱轨迹内的步骤顺序，破坏时序与因果结构。 |
|  |  | `irrelevant` | 用与当前任务无关的轨迹替换。 |
|  |  | `none` / `None` | 不修改。 |
| `insights_modification_type` | **凝练经验**（insights / cheatsheet / memory bank） | `empty` | 清空内容但保留外层模板。 |
|  |  | `corrupt` / `corrupted` | 随机扰乱关键元素（动作引用等），破坏内部一致性。 |
|  |  | `irrelevant` | 替换为通用、与任务无关的摘要。 |
|  |  | `filler_tokens` | 用等长占位字符（如 `%$#@&`）替换全部内容。 |
|  |  | `wo` | 完全删除凝练经验段落（`w/o cond` 全消融）。 |
|  |  | `none` / `None` | 不修改。 |

## 🛠️ 通用环境准备

- Python ≥ 3.10（推荐 3.10/3.11）
- [vLLM](https://github.com/vllm-project/vllm) 用于本地推理服务
- 闭源模型（GPT-4o / GPT-4o-mini / Gemini-2.5-Flash）需要 OpenAI 兼容的 API 接口

运行任何脚本前，先配置好模型凭证：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-openai-compatible-endpoint/v1"
```

各子项目有独立的 `requirements.txt` / 环境依赖，请先按照下文给出的官方仓库链接搭建对应环境，再叠加我们的修改。

---

## 1. ExpeL — 离线单 Agent

> 参考仓库：[LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)

### 环境准备

1. 按照 ExpeL 官方仓库说明搭建 conda 环境，并下载 **ALFWorld**、**HotpotQA**、**WebShop** 数据集。
2. WebShop 任务还需要按照官方说明启动本地 WebShop 服务。
3. 配置 OpenAI 兼容的 `base_url` 与 `api_key`。
4. 安装 ExpeL 依赖：

```bash
cd ExpeL
pip install -r requirements.txt
```

### 三阶段 Pipeline

每个脚本里已经串好了完整流程：

1. **Experience Gathering**（经验收集）—— `train.py` 在训练任务上跑 Agent 并保存原始轨迹。
2. **Insights Extraction**（凝练经验抽取）—— `insight_extraction.py` 从轨迹中提炼 insights。
3. **Faithfulness Evaluation**（忠实性评估）—— `eval.py` 在测试任务上跑 baseline 与各类干预实验。

```bash
cd ExpeL/scripts
bash run_alfworld.sh   # ALFWorld（具身决策）
bash run_hotpot.sh     # HotpotQA（知识问答）
bash run_webshop.sh    # WebShop （网页交互）
```

每个脚本会跑全部 7 种干预：`fewshot ∈ {empty, shuffle, irrelevant}` 与 `insights ∈ {empty, corrupted, irrelevant, filler_tokens}`。日志与 checkpoint 写入 `ExpeL/logs/`。

### 单独调用示例

```bash
python eval.py benchmark=alfworld \
  load_run_name=extracted_insights/insights-extraction-alfworld_14B \
  run_name=my_eval_run \
  agent.llm=Qwen/Qwen3-14B \
  agent.fewshot_strategy=task_similarity \
  agent.faithfulness_experiment=true \
  agent.fewshot_modification_type=shuffle \
  agent.insights_modification_type=none
```

---

## 2. Dynamic Cheatsheet — 在线单 Agent

> 参考仓库：[suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet)

### 环境准备

1. 按照 Dynamic Cheatsheet 官方仓库搭建环境。
2. 配置 OpenAI / Gemini 凭证。
3. 数据集（AIME 2024/2025、GameOf24、GPQA-Diamond、MMLU-Pro Engineering）已包含于上游仓库，位于 `dynamic-cheetsheet/data/`。

### 启动

```bash
cd dynamic-cheetsheet/scripts
bash run.sh         # 跑全部 5 个任务 × 全部干预
bash run_empty.sh   # 仅跑 few-shot 的 "empty" 干预
```

`scripts/run.sh` 会遍历 `tasks=("AIME_2024" "AIME_2025" "GameOf24" "GPQA_Diamond" "MMLU_Pro_Engineering")`，每个任务跑 baseline 加 6 种干预（`insights ∈ {wo, corrupted, irrelevant, filler_tokens}`、`fewshot ∈ {shuffle, irrelevant}`）。

### 单独调用示例

```bash
python run_benchmark.py \
  --task GPQA_Diamond \
  --approach DynamicCheatsheet_RetrievalSynthesis \
  --model_name openai/gpt-4o \
  --save_directory TEST_RESULTS \
  --max_n_samples 100 \
  --faithfulness_experiment \
  --insights_modification_type corrupted \
  --fewshot_modification_type None
```

---

## 3. ReasoningBank — 在线单 Agent（仅凝练经验）

> 参考论文：ReasoningBank (Ouyang et al., 2025)。基于 [WebArena](https://github.com/web-arena-x/webarena) + [BrowserGym](https://github.com/ServiceNow/BrowserGym)。

该框架**只提供凝练经验**而不提供原始轨迹，因此天然适合论文 §4.2 的「仅凝练输入」忠实性分析。

### 环境准备

1. **搭建 WebArena Docker 镜像**——参考 [WebArena 官方指南](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md) 构建或拉取 5 个站点镜像：
   - `shopping_final_0712`
   - `shopping_admin_final_0719`
   - `postmill-populated-exposed-withimg`
   - `gitlab-populated-final-port8023`
   - OpenStreetMap 瓦片服务，也可直接使用公开站点 `https://www.openstreetmap.org`
2. 安装 Python 依赖与 Playwright：
   ```bash
   cd ReasoningBank
   pip install -r requirements.txt
   playwright install chromium
   ```
3. 配置 backbone 凭证。

### 启动

```bash
cd ReasoningBank/scripts
bash run_shopping.sh    # WebArena Shopping
bash run_cms.sh         # WebArena Shopping-Admin (CMS)
bash run_reddit.sh      # WebArena Reddit
bash run_gitlab.sh      # WebArena GitLab    
bash run_map.sh         # WebArena Map        
```

每个脚本依次执行：
1. 构建 memory bank（`--use_bank True`）
2. 关闭 bank 的 baseline（`--use_bank False`）
3. 四种 insights 干预（`wo`、`corrupt`、`irrelevant`、`filler_tokens`）

### 单独调用示例

```bash
python pipeline.py \
  --website shopping \
  --model_name Qwen/Qwen3-32B \
  --faithfulness_experiment True \
  --insights_modification_type corrupt
```

可用参数：
- `--website ∈ {shopping, shopping_admin, gitlab, reddit, map}`
- `--use_bank ∈ {True, False}`——是否使用累积的 memory bank
- `--faithfulness_experiment True --insights_modification_type ∈ {wo, corrupt, irrelevant, filler_tokens}`
- `--faithfulness_experiment True --fewshot_modification_type ∈ {shuffle, irrelevant}`

---

## 4. IG-Score — 信息增益分析

> 基于 [Context Denoising Training](https://github.com/LCM-Lab/context-denoising-training) 代码库。用于论文 §5.2 中定位**抑制检索经验的内部处理偏置**。

### 环境准备

1. 按照 Context Denoising Training 官方仓库安装依赖（`transformers`、`accelerate`、`flash-attn`、`faiss` 等）。
2. 把按变体格式化好的 prompt JSON 文件放入 `cdt/src/inputs/`。脚本期望以下文件：
   - `input_prompt_with.json`
   - `input_prompt_wo.json`
   - `input_prompt_corrupt.json`
   - `input_prompt_irrelevant.json`
   - `input_prompt_empty.json`
   - `input_prompt_filler_tokens.json`

   这些文件可由前述 ExpeL / ReasoningBank / Dynamic Cheatsheet 的运行日志（即原始 prompt 轨迹）导出。

### 启动

仓库提供了按模型规模划分的启动脚本：

```bash
cd cdt/src/scripts
bash run_8B.sh
bash run_14B.sh
bash run_32B.sh
```

每个脚本在所有输入变体上调用 `pipeline_rb.py`：

```bash
python pipeline_rb.py \
  --full_data_path ./inputs/input_prompt_with.json \
  --only_ig True \
  --model_path Qwen/Qwen3-32B
```

结果（逐 token / 逐层的 IG 分数）写入 `cdt/results/`，可使用 `stats_igscore.py` 与 `stats_frscore.py` 进行汇总分析。

---

## 📑 引用

如本工作对您有帮助，请引用：

```bibtex
@inproceedings{zhao2026faithful,
  title     = {Large Language Model Agents Are Not Always Faithful Self-Evolvers},
  author    = {Zhao, Weixiang and Wang, Yingshuo and Zhang, Yichen and Deng, Yang and
               Zhao, Yanyan and Che, Wanxiang and Qin, Bing and Liu, Ting},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 🙏 致谢

本工作基于以下开源项目构建，致以诚挚感谢：

- [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)
- [suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet)
- [WebArena](https://github.com/web-arena-x/webarena) 与 [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [Context Denoising Training](https://github.com/LCM-Lab/context-denoising-training) 
