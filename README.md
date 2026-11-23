# 河南民歌旋律基因编码与掩码扩散项目（Stage 1 ~ Stage 4）

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
  - `model.py`：阶段三双向 Transformer 去噪网络（`MelodyTransformerDenoiser`）
  - `check_stage3.py`：阶段三双向注意力、梯度流、采样闭环与耗时基准验证脚本
  - `train.py`：阶段四正式训练脚本（自动适配 CUDA/MPS/CPU、固定种子验证集、每 10 Epoch 四维体检、断点续训、Loss 与曲线绘图）
- `data/processed/`：已预处理好的标准数据集，可直接开训
  - `train.npy`：`(2390, 64)`，来自 409 首曲目（步长 2 小节）
  - `val.npy`：`(99, 64)`，来自 51 首曲目（步长 8 小节，不重叠）
  - `test.npy`：`(116, 64)`，来自 51 首曲目（步长 8 小节，不重叠）
- `data/samples_stage1/`：阶段一测试集抽样 5 份无损往返验证样片
- `data/samples_stage2/`：阶段二前向渐进掩码预览样片

---

## 快速在 Google Colab GPU 或本地开训（阶段四）

### 方式 A：Google Colab 免费 GPU（约 1 分钟跑完 100 Epoch）
在 Colab 新建笔记本，切换运行时为 **T4 GPU**，依次运行以下单元：

```bash
!git clone https://github.com/LA-ao1/henan-melody-diffusion.git
%cd henan-melody-diffusion
!pip install -q mido
!python3 -m melody_diffusion.train --epochs 120 --batch-size 64
```

### 方式 B：本地电脑（Windows / Mac / Linux）
```bash
git clone https://github.com/LA-ao1/henan-melody-diffusion.git
cd henan-melody-diffusion
pip install -r requirements.txt

# 启动训练（自动选择 CUDA / Apple MPS / CPU）
python3 -m melody_diffusion.train --epochs 120 --batch-size 64

# 如中途中断，可从最近检查点续训：
python3 -m melody_diffusion.train --epochs 120 --resume checkpoints/last_model.pt
```

### 训练产物说明（保存于 `checkpoints/`）
- `best_model.pt`：验证集 Loss 最低的最优权重（供阶段五实验加载）
- `last_model.pt`：最新一轮权重（支持 `--resume` 断点续训）
- `training_history.json`：完整训练/验证 Loss 与每 10 轮四维体检指标记录
- `training_curves.png`：Loss 收敛曲线与五声纯度/非法 `HOLD` 演变曲线
- `checkup_samples/`：每 10 轮自动采样的生成序列 `.mid`、`.wav` 音频与简谱对照 `.txt`
