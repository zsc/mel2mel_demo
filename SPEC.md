下面是一版可直接保存为 `SPEC.md` 的内容：

# SPEC: 用图像编辑模型学习 mel 频谱修改，以实现近似 `wav -> wav` 编辑

## 1. 项目目标

构建一个训练与推理流程，使一个**图像编辑模型**能够学习把 `source mel` 修改为 `target mel`。
由于 mel 频谱严格遵循某个 **neural vocoder** 的定义，因此编辑后的 mel 可以被该 vocoder 还原成 waveform，最终实现近似的 `wav -> wav` 编辑。

v1 的核心目标不是做通用语音生成，而是验证以下命题：

> 给定同一句文本的两个版本：
>
> * 一个来自较弱或机械化的 TTS/前端（如 `espeak`）形成的 `source`
> * 一个来自高质量 neural TTS（如 `qwen-tts`）形成的 `target`
>
> 是否可以训练一个图像编辑模型，把 `source mel` 转换成更接近 `target mel` 的 mel，并在 vocoder 后获得更自然的语音。

---

## 2. 成功标准

项目在 v1 中被视为成功，当满足以下条件：

1. 能稳定生成成对数据：

   * `text`
   * `source wav`
   * `target wav`
   * `source mel`
   * `target mel`
   * `aligned_source mel`（可选，按概率生成）
   * metadata / manifest

2. 能训练一个 paired image-editing baseline，使模型输入 `source mel` 后输出接近 `target mel` 的结果。

3. 输出 mel 经过固定 vocoder 还原为 wav 后：

   * 内容可懂度不明显劣化
   * 听感明显优于 `espeak` 直接生成的源音频
   * 客观指标相对 `source` 有可测提升（如 mel L1 / MR-STFT / ASR CER）

---

## 3. 非目标

以下内容不在 v1 范围内：

* 多语言混训
* 多说话人建模
* 实时流式推理
* 说话人克隆
* 完整替代端到端 TTS
* 与所有 vocoder 兼容

v1 只做**单语言、单 mel 定义、单 vocoder、单高质量 TTS 后端**。

---

## 4. 实现前必须锁定的常量

以下配置一旦确定，整个数据集和训练过程都必须保持一致，不允许混用：

### 4.1 语言范围

* `LANGUAGE = <one language only>`
* 建议 v1 只做一种语言，不要混语种。

### 4.2 Vocoder 规范

必须先锁定一个固定的 neural vocoder，并把 mel 定义视为**全项目唯一标准**。

必填配置包括：

* `sample_rate`
* `n_fft`
* `win_length`
* `hop_length`
* `n_mels`
* `fmin`
* `fmax`
* 幅度表示方式（log-mel / natural log / log10）
* 归一化方式
* 动态范围裁剪方式

**硬性要求**：
所有 `source wav` / `target wav` / 推理输出 wav，都必须通过**同一套 mel 定义**处理。
**禁止**混入不同 TTS / vocoder 自带的 mel 配置。

### 4.3 高质量 TTS 后端

* 默认候选：`qwen-tts`
* 需要抽象成可替换后端接口：`synthesize_target(text) -> wav`

### 4.4 弱前端 / 源端生成器

* 默认候选：`espeak`
* 需要抽象成可替换后端接口：`synthesize_source(text) -> wav`

---

## 5. 数据规模假设

### 5.1 初始规模

* v1 先收集 `10,000` 个句子

### 5.2 对“1 万句是否够”的结论

`10,000` 句**足够作为可行性验证（POC / MVP）**，但**不太够作为最终稳定方案**。

原因：

* 图像编辑模型需要学习从机械 TTS 到高质量 TTS 的谱面映射
* 除了文本内容，还要覆盖：

  * 音素组合
  * 韵律变化
  * 标点停顿
  * 长短句
  * 数字、英文、符号等边界情况
* 如果平均每句时长不长，`10,000` 句通常只够做第一轮验证，不足以获得强泛化

### 5.3 数据规模建议

* `10k`：做通路验证、损失曲线、主观试听、小规模 ablation
* `30k~100k`：更适合做稳定训练和泛化评估

### 5.4 重要结论

对 v1 来说，**音素覆盖和句型多样性比句子条数更重要**。
如果只能做 1 万句，应优先保证：

* 音素平衡
* 长短句均衡
* 标点分布合理
* 包含数字、专名、英文夹杂等困难样本

---

## 6. 数据构建方案

## 6.1 文本语料收集

收集约 `10,000` 个句子，要求：

* 去重
* 文本合法、可正常送入两个 TTS 后端
* 尽量覆盖常见音素组合
* 句长分布合理
* 过滤过短和过长句
* 过滤异常符号、乱码、极端重复字符
* 明确授权或使用可合法使用的语料

每条样本至少包含：

* `utt_id`
* `text`
* `language`
* `split`（train/val/test）

---

## 6.2 target 数据生成

对每条 `text`：

1. 使用高质量 TTS（如 `qwen-tts`）合成 `target wav`
2. 使用**固定 vocoder 定义对应的 mel 提取器**从该 wav 提取 `target mel`

输出：

* `target_wav.wav`
* `target_mel.npy`

要求：

* target 端采样率统一
* 提取 mel 前完成必要 resample
* 记录 wav 时长、帧数、是否成功、失败原因

---

## 6.3 source 数据生成

对同一条 `text`：

1. 使用 `espeak` 合成 `source wav`
2. 使用同一 mel 提取器提取 `source mel`

输出：

* `source_wav.wav`
* `source_mel.npy`

要求：

* source 与 target 的文本完全一致
* mel 配置与 target 完全一致
* 不做任意图像式增强（禁止颜色增强、翻转、旋转）

---

## 6.4 DTW 对齐增强

为了减轻 source/target 在时长和局部节奏上的差异，可对 `source mel` 以 `p = 0.5` 的概率做一次 DTW 对齐，生成 `aligned_source mel`。

### 规则

* 以 `50%` 概率：

  * 对 `source mel` 和 `target mel` 做 DTW
  * 根据对齐路径把 `source mel` warp 到更接近 `target` 的时间轴
* 以 `50%` 概率：

  * 保留原始 `source mel`
  * 不做对齐

### 目的

* 一部分样本保留真实时长差异，帮助模型学习“从差时长到自然谱面”的映射
* 一部分样本减轻时间错位，降低训练难度

### 实现要求

* DTW 必须是单调、不可回退的对齐
* 保存对齐路径或可复现 metadata
* 记录 `use_dtw = true/false`
* 记录对齐前后帧数

### 风险提示

DTW 可能引入不自然的时间拉伸，甚至把错误频段硬对齐。
因此 v1 必须做三组对比实验：

1. `no_dtw`
2. `all_dtw`
3. `mixed_50p_dtw`（默认方案）

---

## 7. 训练样本定义

每个训练样本至少包含以下字段：

* `utt_id`
* `text`
* `source_wav_path`
* `target_wav_path`
* `source_mel_path`
* `target_mel_path`
* `use_dtw`
* `aligned_source_mel_path`（若使用）
* `num_source_frames`
* `num_target_frames`
* `duration_sec`
* `split`

训练时真正喂给模型的输入定义为：

* 输入：`editor_input_mel`

  * 若 `use_dtw = true`，则为 `aligned_source_mel`
  * 否则为 `source_mel`
* 目标：`target_mel`

---

## 8. mel 作为图像的表示规范

由于要使用图像编辑模型，必须把 mel 视为“受约束的单通道图像”，但不能按自然图像随意处理。

### 8.1 表示方式

* 默认使用单通道 log-mel
* 若 backbone 强制要求 3 通道，则复制成 3 通道，不做伪彩色映射

### 8.2 归一化

* 将 mel 映射到固定数值范围（如 `[0, 1]` 或 `[-1, 1]`）
* source / target / inference 必须使用同一归一化方式

### 8.3 几何限制

禁止以下操作：

* 水平翻转
* 垂直翻转
* 旋转
* 任意仿射变换
* 颜色抖动
* 裁掉频率轴

### 8.4 可变长度处理

语音长度可变，因此训练必须支持以下之一：

1. **固定窗口切块**

   * 沿时间轴切成固定帧长窗口
   * 对长句切块、对短句补 pad

2. **bucketing + pad**

   * 同批次按相近长度分桶
   * 只在时间轴右侧 pad

v1 推荐先用：

* 固定 `n_mels`
* 固定时间窗口
* 长句滑窗切块
* 短句右侧 pad

**禁止**对 mel 做各向同性 resize，尤其不能压缩/拉伸频率轴。

---

## 9. 模型范围

v1 不限定具体图像编辑 backbone，但要求它支持**paired image-to-image learning**。

### 最低要求

模型必须支持：

* 输入一张“源 mel 图”
* 输出一张“目标 mel 图”
* 用 paired supervision 训练

### 可选条件

* 可以额外接收 `text` 作为条件
* 但 v1 不是必须，因为 source/target 文本相同

### 推荐策略

先做一个最小 baseline：

* 仅使用 `source mel -> target mel` 的 paired 监督
* 暂不引入复杂 prompt engineering
* 文本只作为 metadata 保留，供后续扩展

---

## 10. 训练与实验计划

v1 至少完成以下实验：

### 实验 A：基础可行性

* 数据：`10k` 句
* 输入：raw `source mel`
* 目标：`target mel`

### 实验 B：全量 DTW

* 数据：`10k`
* 所有样本都使用 DTW 后的 `aligned_source mel`

### 实验 C：50% DTW 混合

* 数据：`10k`
* 按 `p=0.5` 使用 DTW
* 这是当前默认主方案

### 实验 D：小样本 sanity check

* 数据：`500~1000` 句
* 用于快速验证整条 pipeline 可训练、可过拟合、可生成 demo

---

## 11. 评估指标

至少实现以下评估：

### 11.1 谱面距离

* mel L1 / L2
* log-mel distance

### 11.2 波形级重建质量

将预测 mel 送入固定 vocoder 后，与 `target wav` 比较：

* multi-resolution STFT loss
* 其他可实现的感知质量指标

### 11.3 内容保持

对重建 wav 做 ASR，比较其转写与原始 `text`：

* CER / WER

### 11.4 主观试听

至少导出一组样例：

* `source wav`
* `target wav`
* `pred wav`

要求能人工判断：

* 是否更自然
* 是否保留原文本内容
* 是否比 source 更接近 target

---

## 12. 验收标准

v1 的最低验收标准：

1. 数据生成脚本可自动跑完整语料，并输出 manifest
2. 训练脚本可在小样本集上正常收敛
3. 在验证集上，相比 `source -> target` 基线，预测结果至少满足：

   * 谱面距离下降
   * 语义可懂度不显著恶化
4. 能导出不少于 20 组可试听样例
5. 代码支持复现：

   * 固定随机种子
   * 固定 split
   * 固定 mel config
   * 固定对齐策略

---

## 13. 代码交付物

Codex 需要产出以下模块：

### 13.1 数据准备

* 文本清洗与去重
* train/val/test 划分
* manifest 生成

### 13.2 音频生成

* 高质量 TTS 后端封装
* `espeak` 后端封装
* 失败样本重试与日志

### 13.3 mel 提取

* 统一 mel 提取器
* 严格对齐 vocoder 配置

### 13.4 DTW 模块

* `source mel` 与 `target mel` 的 DTW 对齐
* warp 后 mel 导出
* 对齐 metadata 保存

### 13.5 训练模块

* 数据加载器
* 可变长度处理
* baseline image editor 训练脚本
* checkpoint / eval / demo 导出

### 13.6 推理模块

* 输入任意 `source wav`
* 转 mel
* 送入 editor
* 输出 `edited mel`
* 用固定 vocoder 还原 `pred wav`

### 13.7 评估模块

* 客观指标计算
* demo 样本导出
* ablation report

---

## 14. 建议的目录结构

* `SPEC.md`
* `configs/`

  * `mel.yaml`
  * `train.yaml`
  * `data.yaml`
* `scripts/`

  * `build_corpus.py`
  * `synthesize_target.py`
  * `synthesize_source.py`
  * `build_pairs.py`
  * `run_dtw_align.py`
  * `train_editor.py`
  * `infer_editor.py`
  * `evaluate.py`
* `src/`

  * `audio/`
  * `tts/`
  * `alignment/`
  * `dataset/`
  * `models/`
  * `vocoder/`
  * `metrics/`
* `data/`

  * `raw_text/`
  * `wavs/`
  * `mels/`
  * `manifests/`
* `outputs/`

  * `checkpoints/`
  * `samples/`
  * `reports/`

---

## 15. 关键风险

### 15.1 mel 不是普通图像

图像模型常见的平移不变性、颜色增强假设不一定适用于 mel。
频率轴尤其不能按自然图像处理。

### 15.2 DTW 可能过度“纠正”

如果 source 和 target 的声学风格差异太大，DTW 可能学到错误对齐。
因此必须做 ablation，而不是直接假定 `50% DTW` 一定最优。

### 15.3 eSpeak 的语言质量可能太差

如果 `espeak` 在目标语言上的发音过差，模型可能更像在“修复错误发音”而不是做自然度编辑。
这会增加任务难度，也可能污染对齐。

### 15.4 1 万句的泛化能力有限

1 万句更适合作为第一阶段验证。
如果结果可行，应尽快扩到更大语料。

---

## 16. v1 的推荐决策

若无额外指示，默认采用以下设置：

* 单语言
* `10k` 句作为 MVP
* `qwen-tts` 生成 `target wav`
* `espeak` 生成 `source wav`
* 用固定 vocoder 配置提取 mel
* `50%` 概率做 DTW 对齐
* 先训练 paired image-to-image baseline
* 不做文本条件
* 先验证可行性，再决定是否扩大到 `30k+`

---

## 17. 最终一句话定义

v1 要交付的是一个**可复现的 paired mel-editing pipeline**：

> 用同一句文本生成低质量 `source mel` 和高质量 `target mel`，
> 训练图像编辑模型学习 `source mel -> target mel` 的映射，
> 再通过固定 neural vocoder 把编辑后的 mel 还原成更自然的 wav。
