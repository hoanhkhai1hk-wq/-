# RGB 训练演练（TRN-20261016-001）

本页说明 `scripts/rgb/train_smoke.py` 的两个命令。它只做**训练代码的机械流程演练**，
**不是**用上传的农田照片训练模型，也**不是**最终参赛模型。

> 一句话：真实数据只"读一读、算算张量"；真正的"前向→损失→反向→更新参数→保存→重载"
> 只在 4 张 32×32 的**内部合成小图**上跑有限几步，用来证明代码能跑通。

## 先弄清两个概念

- **真实候选**：`reports/labels/RGB-candidates-DATA-20261016-005` 里的 855 张田块图＋掩膜。
  它们仍待人工核对语义、别名与来源，`semantic_reviewed=false`、`not_training_ready=true`，
  **不具备正式训练条件**。本工具只**只读**它们，绝不喂给模型或优化器。
- **合成样本**：程序运行时临时画出的彩色几何块小图（含忽略区）。它们**不是**真实农田，
  只用于软件演练。演练网络是随机初始化的小网络，**不替代、不修改现有 FTW**，
  也**不预选**最终农业模型。

## 环境

用现有解释器，不需安装、不联网：

```
.venv-recognition/Scripts/python.exe
```

已含 Torch 2.14.1+cpu、Pillow、Numpy；**无 CUDA**。不要用 `.venv-p0`（它没有 Torch）。

**相对路径基准**：命令行里 `--input` / `--output` 若写成相对路径，一律**相对项目根**
（脚本所在仓库的根，即 `D:\AI_Workspace\Tasks\兴智杯`）解析，**不是**相对当前 shell 的
工作目录。因此从任意目录运行、或使用相对路径都一致。下面的命令以 Windows PowerShell
书写（续行用反引号 `` ` ``，不要用 cmd.exe 的 `^`）。

## 命令一：`smoke`（合成演练）

在 4 张 32×32 合成图与 0/1/2/255 掩膜上，经历 DataLoader 组 batch，执行有限步
CPU 前向/损失/反向/参数更新，保存并重载随机小网络。

```powershell
& .venv-recognition/Scripts/python.exe -B scripts/rgb/train_smoke.py smoke `
    --output reports/rgb/TRN-20261016-001-smoke-manual --steps 12 --seed 17
```

- **输入**：无真实数据；程序自行生成合成 PNG。
- **操作**：CLI 会在 `reports/rgb` 下创建一个**全新**目录（已存在即拒绝，不覆盖）。
- **结果**：目录里有 `summary.json`、`synthetic_state.pt`、`synthetic/`（合成 PNG）与 `README.md`。
  `summary.json` 关键字段：`kind=synthetic_smoke`、`optimizer_steps`（真实步数）、
  `parameter_changed`、`checkpoint_roundtrip_equal`、`initial_loss`、`final_loss`、`seed`、
  以及输入/输出形状与 dtype。loss **不保证每步下降**。
- **loss 口径**：`initial_loss` 与 `final_loss` 都在**同一个固定合成参考 batch**上计算
  （`shuffle=False` 的前 2 张合成图，`reference_batch_indices=[0,1]`，
  `loss_reference='same_fixed_synthetic_batch_before_after'`）：`initial_loss` 在**任何更新前**、
  `final_loss` 在**全部更新后**各实算一次；`final_loss` 可用保存的 checkpoint 在同样输入上独立复算。
  `last_step_loss_before_update` 另记最后一步更新前的 batch loss，供参考。
- **有限性检查**：每次反向都会检查 loss、全部参数梯度与更新后参数是否有限；发现 Inf/NaN、
  参数未发生变化或 checkpoint 严格重载后输出不一致，都会显式报错，**不会**返回带非有限权重
  或未更新的"成功"结果。失败目录保留，不删除。
- **seed**：`--seed` 必须是 `0..2**32-1` 的整数（非 bool）；负数或超范围会在建目录前被拒绝。
- **下一步**：这只证明代码流程可用；真正训练要先完成人工修标、别名与来源确认、防泄漏划分。

## 命令二：`inspect`（真实候选只读检查）

遍历**所有** candidate（实际数量按 manifest，不硬编码 855），逐张构造并校验张量，
汇总每类像素与忽略像素。

```powershell
& .venv-recognition/Scripts/python.exe -B scripts/rgb/train_smoke.py inspect `
    --input reports/labels/RGB-candidates-DATA-20261016-005 `
    --output reports/rgb/TRN-20261016-001-inspect-manual
```

- **输入**：005 候选目录（须在 `reports/labels` 或 `reports/rgb` 内）。
- **操作**：先解析并检查全新的输出位置（**不创建**），再只读源图与掩膜，校验 SHA/字节/尺寸/RGB/值域；
  **不建模型、不建优化器、不反向**。输出不得位于输入目录内。全部记录成功后，才创建 `reports/rgb`
  下全新输出目录。
- **结果**：`summary.json`（含 `kind=real_input_inspection`、`optimizer_steps=0`、
  `trained_on_real_data=false`、`semantic_reviewed=false`、`training_ready=false`、
  `split_created=false`、`excluded_review_count`、各类像素总量，以及 `input_dir`（相对项目根）、
  `manifest_sha256`、`policy_sha256`）与 `records.json`（逐图，含来源字段
  `image_path`/`image_sha256`/`json_path`/`json_sha256`/`mask_path`/`mask_sha256`；
  源图与 JSON 路径相对**项目根**，`mask_path` 相对**输入目录**）。
- **下一步**：这些像素统计**不是精度、不是真值、不是实际面积**；候选语义仍待人工确认。

## 常见问题

- **没有"真实训练"命令**：CLI 只有 `smoke` 和 `inspect`；也没有强制绕过 `not_training_ready`
  的开关。这是刻意设计的，防止把未审数据当训练集。
- **无参数运行**：只显示帮助，不执行任何动作。
- **坏参数**：`--steps` 必须是 1..100 的整数（非 bool），`--seed` 必须是 0..2**32-1 的整数
  （非 bool）；否则以非零退出码和中文错误拒绝。
- **输出目录**：必须在 `reports/rgb` 下的**全新**目录；已存在、位于 `reports/labels`、位于
  输入目录内、或项目根一律拒绝；不删除、不覆盖。
- **子进程编码**：测试用 `encoding='utf-8', errors='strict'` 并显式设置
  `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1`/`PYTHONDONTWRITEBYTECODE=1`，避免 Windows 默认
  GBK 读取中文输出时报 `UnicodeDecodeError`。

## 参考（Codex 已核对的官方资料）

- CrossEntropyLoss（`ignore_index`）：
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
- torch.load（`weights_only`）：
  https://docs.pytorch.org/docs/2.14/generated/torch.load.html

## 仍未完成（不属本工具范围）

- 人工修正 48 份几何问题副本并重新质检。
- 确认 `Fieldridge` 别名口径、全体标注语义/忽略区规则。
- 确认 `邓` 与 `hy0713 (1)` 两批的来源空间/时相关系，制定防同源泄漏的训练/验证/测试划分。
- 现 8 波段 FTW 与 RGB 不兼容；无独立沟渠类别；不得用田埂替代沟渠。

以上完成前，真实数据仍**不划分、不训练、不评精度、不算平方米**。
