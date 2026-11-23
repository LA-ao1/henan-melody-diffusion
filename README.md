# 河南民歌旋律基因编码与掩码扩散项目（Stage 1 & Stage 2）

小组期中研究课题：**3 旋律的扰动与重建**  
技术路线：**河南民歌 `2/4` 子集 + 十六分音符时间网格（每小节 8 步，`L = 64`） + 掩码扩散模型（Masked Diffusion）**

---

## 小组必读文档

- [课题总览、原理科普与分工指南 (`docs/midterm_project_guide.md`)](docs/midterm_project_guide.md)
- [模型训练与实验操作步骤指南 (`docs/training_workflow.md`)](docs/training_workflow.md)

---

## 仓库目录结构

- `docs/`：课题说明文档与训练步骤手册
- `melody_diffusion/`
  - `codec.py`：阶段一双向十六分音符基因编解码器（支持 MusicXML 解析与 MIDI / MusicXML / WAV 还原导出）与音乐学统计指标函数
  - `prepare_henan.py`：阶段一数据过滤、按原曲 ID 防泄漏划分、切片编码与 100% 无损往返校验脚本
  - `diffusion.py`：阶段二掩码扩散核心模块（`HenanMelodyDataset`、`MaskedDiffusion` 前向加噪 $\gamma_t = t/T$、掩码交叉熵损失、精确马尔可夫后验 `1/t` 反向采样）
  - `preview_stage2.py`：阶段二单元测试与前向渐进掩码预览导出脚本
- `data/processed/`：已预处理好的标准数据集，可直接用于阶段二~四训练
  - `train.npy`：`(2390, 64)`，来自 409 首曲目（步长 2 小节）
  - `val.npy`：`(99, 64)`，来自 51 首曲目（步长 8 小节，不重叠）
  - `test.npy`：`(116, 64)`，来自 51 首曲目（步长 8 小节，不重叠）
  - `splits_metadata.json`：每条窗口所属原曲 ID、曲名、类别及起始小节索引
  - `dataset_stats.json`：全库统计指标与校验报告
- `data/samples_stage1/`：阶段一测试集抽样 5 份无损往返验证样片（含 `.mid`、`.musicxml`、`.wav` 及 `.txt`）
- `data/samples_stage2/`：阶段二前向渐进掩码预览样片（`t = 0, 5, 15, 25, 35, 45, 50` 对应 `.mid`、`.wav`、`.txt` 及 `stage2_verification.json`）

---

## 整数编码词表（`V = 40`, `L = 64`）

| Token ID | 含义 | 说明 |
| :--- | :--- | :--- |
| `0` | `REST` | 休止符（无声） |
| `1` | `HOLD` | 延长符号（延续上一时间步音高） |
| `2 ~ 38` | `PITCH` | `MIDI 48 (C3) ~ 84 (C6)`（`token = midi - 46`，覆盖 99.96% 河南 `2/4` 音符） |
| `39` | `[MASK]` | 掩码扩散专用吸收态掩码符号 |

---

## 本地运行与复现说明

如只需直接开始阶段二/三开发，仓库内的 `data/processed/*.npy` 已准备好，无需重新下载原始压缩包。

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 复现阶段二前向加噪检验与掩码样片生成：
   ```bash
   python3 -m melody_diffusion.preview_stage2
   ```
3. （可选）从原始数据重跑阶段一：从 [Anthology-of-Chinese-Folk-Songs (`v261104`)](https://github.com/m-july/Anthology-of-Chinese-Folk-Songs/releases/tag/v261104) 下载 `henan.jsonl` 与 `ACFS-v261103-scores-henan.zip` 放入 `data/raw/`，运行：
   ```bash
   python3 -m melody_diffusion.prepare_henan
   ```
