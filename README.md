# 河南民歌旋律基因编码与掩码扩散项目（全流程完成）

小组期中研究课题：**3 旋律的扰动与重建**  
技术路线：**河南民歌 `2/4` 子集 + 十六分音符时间网格（每小节 8 步，`L = 64`） + 掩码扩散模型（Masked Diffusion）**

---

## 给小组成员：一键下载与试用模型

1. **直接下载整套仓库、已训模型与阶段五实验图表/音频包**：
   - 点击直接下载压缩包：[henan-melody-diffusion-main.zip](https://github.com/LA-ao1/henan-melody-diffusion/archive/refs/heads/main.zip)
   - 或命令行克隆：
     ```bash
     git clone https://github.com/LA-ao1/henan-melody-diffusion.git
     cd henan-melody-diffusion
     ```
2. **在浏览器直接看“全流程交互式图解与试听”**：
   - 解压后双击打开 [docs/pipeline_explorer.html](docs/pipeline_explorer.html) 即可在浏览器直观查看 **阶段一 ~ 阶段四 每一步在做什么**，并直接试听原曲、掩码破坏与模型生成/重建旋律。
3. **在本地具体试用已训好的模型**：
   ```bash
   pip install -r requirements.txt

   # 方式 A：交互式引导菜单（直接按提示选 1 或 2）
   python3 -m melody_diffusion.try_model

   # 方式 B：命令行直接生成 3 首新民歌（导出至 outputs/try/）
   python3 -m melody_diffusion.try_model --mode generate --num 3 --temperature 0.9

   # 方式 C：把测试集第 29 首民歌掩码损坏 50% 后让模型重建对比
   python3 -m melody_diffusion.try_model --mode reconstruct --song-idx 29 --mask-pct 50
   ```

---

## 小组必读文档

- [全流程交互式图解网页 (`docs/pipeline_explorer.html`)](docs/pipeline_explorer.html)
- [课题总览、原理科普与分工指南 (`docs/midterm_project_guide.md`)](docs/midterm_project_guide.md)
- [模型训练与实验操作步骤指南 (`docs/training_workflow.md`)](docs/training_workflow.md)

---

## 仓库目录结构

- `docs/`：课题说明文档、训练步骤手册与交互式可视化图解网页
- `melody_diffusion/`
  - `codec.py`：阶段一双向十六分音符基因编解码器（支持 MusicXML 解析与 MIDI / MusicXML / WAV 还原导出）与音乐学统计指标函数
  - `prepare_henan.py`：阶段一数据过滤、按原曲 ID 防泄漏划分、切片编码与 100% 无损往返校验脚本
  - `diffusion.py`：阶段二掩码扩散核心模块（`HenanMelodyDataset`、`MaskedDiffusion` 前向加噪 $\gamma_t = t/T$、掩码交叉熵损失、精确马尔可夫后验 `1/t` 反向采样）
  - `preview_stage2.py`：阶段二单元测试与前向渐进掩码预览导出脚本
  - `model.py`：阶段三双向 Transformer 去噪网络（`MelodyTransformerDenoiser`）
  - `check_stage3.py`：阶段三双向注意力、梯度流、采样闭环与耗时基准验证脚本
  - `train.py`：阶段四正式训练脚本（已完成 100 Epoch，最优权重已上传至 `checkpoints/best_model.pt`）
  - `try_model.py`：供小组成员本地直接试听与试用已训练模型的交互/命令行脚本
  - `evaluate.py`：阶段五全量测试集（116 首）梯度掩码重建实验、4 大音乐学现象统计绘图与典型案例导出脚本
- `checkpoints/`：已完成 100 轮训练的模型产物（最佳 `val_loss = 1.2426` @ Epoch 90）
- `results/stage5/`：阶段五全套可直接用于研究报告的图表、JSON 统计与 3 首典型曲目乐谱/音频包
  - `stage5_metrics_curves.png`：四大音乐学现象（调式吸引子 / 轮廓 / 节奏 / 动机）随噪声演化曲线
  - `stage5_pitch_histogram.png`：十二半音音高分布直方图（验证五声骨干音吸引子效应）
  - `stage5_case_studies.png`：典型测试曲梯度重建钢琴卷帘对比图
  - `stage5_results.json`：完整量化实验指标表
  - `cases/`：3 首典型测试曲在多档噪声下的 `.mid`、`.musicxml`、`.wav` 与简谱对比 `.txt`
- `data/processed/`：已预处理好的标准数据集（`train.npy` `2390×64`、`val.npy` `99×64`、`test.npy` `116×64`）
- `data/samples_stage1/` & `data/samples_stage2/`：阶段一与阶段二验证样片

---

## 复现阶段五实验命令

```bash
python3 -m melody_diffusion.evaluate
```
