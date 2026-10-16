# RGB 候选数据准备（DATA-20261016-004 / 005 修复）

本模块把已标注的 RGB 资料机械转换为**候选掩膜**与**待人工复核副本**。它只读原始文件，
所有产物都是可重建的派生结果，**未经语义复核、未训练就绪**。

> 005 在 004 初版基础上修边界问题（见下“边界修复”），不重放 004，也未改动任何原标注/图像。

## 什么时候用

在完成既有两批标注资料检查后，需要：

- 把 `Field / Field ridge / Fieldridge / Road` 统一成机器可读的类别掩膜，供后续复核与训练准备；
- 把存在几何问题的标注（自交、画布越界）单独挑出来，生成可人工编辑的副本，**先修再看**。

本工具不判断标注语义正确与否，也不划分训练/验证集。

## 类别与像素规则

| 值 | 含义 |
|---|---|
| 0 | Field 田块 |
| 1 | Field ridge 田埂（含派生归一的 `Fieldridge` 拼写变体） |
| 2 | Road 道路 |
| 255 | 忽略：未画区域、不同类别重叠>1、未确认即不可用 |

- **0 不是背景**；本项目不设背景类别。未画区域一律 255，不当作已知背景。
- `Fieldridge` → `Field ridge` 只是**派生层的拼写归一假设**（待语义复核），原 label 与修改记录都保留。
- 无独立沟渠类别；田埂不得直接改称沟渠。
- 满幅不同类别叠加（整图 255、有效互斥像素为 0）记为 **empty 诊断图**，不是候选。

## 转换策略（写入 policy.json）

1. 先做**完整输入校验，且全部在创建输出目录之前完成**：全部原始文件 SHA256/字节数与索引一致、
   `sample_id` 唯一且为便携安全组件 `[A-Za-z0-9_-]{1,80}`（拒绝 Windows 保留名、拒绝仅大小写
   不同的重复 id）、来源按 resolve+normcase 去重（同一 JSON/图像不得重复引用）、
   JSON `imagePath` **严格等于**实际 TIFF basename、图像为 RGB 且尺寸与 JSON/索引一致；
   JSON 宽高必须为正整数（拒绝 bool/非正）；类型不接受 bool、坏 SHA、非有限坐标。
   **完整索引与任一（含最后一个）来源出错 → 非零退出，且不产生输出目录（无半成品）。**
2. 逐 shape 几何检查：至少 3 个不同有限点；画布边界 `[0,W]/[0,H]`（EPS=1e-6）；
   非相邻边严格内部穿越（orientation 阈值 1e-6 px²，双方向一致，参考已验证算法）。
   任一份标注含自交、越界或点数不足 → **整份进入 review，不生成训练 mask**。
   只重复点/显式闭合点不自动判错，**不重排、不改写点序、不追加或回写任何源坐标**；
   Pillow 填充时对多边形做**隐式闭合**（自动连接末点与首点）。
3. 其余文件机械转换：按现有 Pillow `ImageDraw.polygon` 填充，输入坐标原样（含 float）；
   同类取并集，不同类别覆盖>1 一律 255，未画 255。结果与绘制顺序无关。
   输出 L 模式原尺寸 mask PNG，唯一值限 `0/1/2/255`。
   **有效像素判定只取互斥类别像素（`class_pixels`）之和**，不含跨类冲突/未画；为 0 记为不可用（empty）。

## 输出目录结构

| 文件 | 说明 |
|---|---|
| `policy.json` | 全部策略与 schema、别名假设、无背景类别、栅格化规则、路径基准、Pillow 版本 |
| `manifest.json` | 全部记录：源路径/SHA/bytes、原→派生类别映射、状态、掩膜/副本路径与 SHA、像素统计、复核原因 |
| `masks/<sample_id>.png` | 候选索引图（值域 0/1/2/255；empty 也保留诊断图） |
| `review/<sample_id>.png` | 与原 TIFF 逐像素一致的可编辑副本 |
| `review/<sample_id>.json` | points 原样保留，`imageData` 置空，`imagePath` 指向同目录 PNG，Fieldridge 拼写归一；逐 shape 变更记录在 manifest |
| `review.csv` | UTF-8 BOM；source_id/源目录/文件/shape 问题/副本路径/待复核状态 |
| `summary.json` | 实际输入/候选/review/零有效计数与像素、类别统计 |
| `README.md` | 零基础说明 |

状态取值为 `candidate / review / empty / unsupported`。`unsupported` 表示存在未知 label 或
非 polygon shape，整份排除并记录，不默当背景。

### 两种路径基准

- **源路径**（`manifest.json` 的 `json_path`/`image_path`）相对**项目根**。
- **派生产物路径**（`mask_path`、`review_png_path`、`review_json_path` 及 `review.csv` 中的
  `review_png`/`review_json`）相对**本输出目录**，形如 `masks/<id>.png`、`review/<id>.png`。
- 把本输出目录整体拷贝走后，源路径仍需回到项目根解释，派生产物路径相对本目录解释。
  该约定同时写入 `manifest.json`/`policy.json`/`summary.json` 的 `path_bases`。

## 运行方式（由 Codex 执行）

```powershell
& .venv-p0/Scripts/python.exe -B scripts/labels/prepare_candidate.py
```

> 使用现有 `.venv-p0` 解释器（`-B` 不写字节码）；**不安装、不新增、不改动环境**。

- 默认输入 `<根>/reports/labels/DATA-20261016-004-input.json`（005 沿用 004 索引），
  默认输出 `<根>/reports/labels/` 下全新目录。
- `--input` 自定义索引必须位于 `reports/labels` 内；其引用的原始文件必须落在 `邓/`、
  `hy0713 (1)/` 或 `reports/labels/`（fixture）内，拒绝绝对路径与 `..` 越界。
- `--output` 只允许 `reports/labels` 内**全新**目录；已存在即拒绝，不覆盖、不删除。

## 测试（由 Codex 运行）

```powershell
& .venv-p0/Scripts/python.exe -B -m unittest tests.labels.test_candidate -v
```

测试只写 `reports/labels/` 下全新 fixture 目录，不使用自动删除的临时目录；
子进程显式 UTF-8、`-B`、`PYTHONDONTWRITEBYTECODE=1`。回归覆盖：危险/保留 `sample_id`、
大小写折叠重复 id、来源 resolve+normcase 去重、末来源坏 SHA 不出输出目录、imagePath 同 stem
异后缀/外来前缀、JSON 宽高 bool/非正、互斥有效像素（满幅跨类叠加为 empty）、同类并集独立手算
52、派生产物路径相对输出目录、policy 闭合措辞等。

## 边界修复（005）

1. `sample_id` 先用便携安全组件校验（`[A-Za-z0-9_-]{1,80}`，拒绝 Windows 保留名、大小写折叠重复），
   再用于文件名，杜绝 `../../escape` 之类逃逸；测试用纯校验与 `prepare` 双路径覆盖，未向 fixture 外写文件。
2. 完整索引与**所有**来源的 SHA/bytes/解析/属性/坐标检查，全部移到**创建输出目录之前**；末来源坏图/坏
   SHA 也抛 `PrepareError` 且输出目录不存在（两遍读取，不把 903 条内嵌 `imageData` 常驻内存）。
3. 有效像素改用互斥 `class_pixels` 之和；满幅跨类叠加全 255 记为 empty，非 candidate（诊断 mask 保留）。
4. JSON `imagePath` 必须严格等于实际 TIFF basename；JSON 宽高必须为正整数（拒绝 bool）。
5. `manifest.json`/`review.csv` 派生产物路径改为相对输出目录；源路径仍相对项目根，并在
   manifest/policy/文档写明两种基准。
6. 修正 `test_same_class_overlap_is_union` 的错误预期（原 `<50`），改独立手算 52，未改栅格化实现。
7. policy “不闭合”歧义改为“不改写点序/不追加源坐标，Pillow 填充时隐式闭合多边形”。

## 边界与未完成

- 候选/副本**不等于**语义已审核；全部标记 `semantic_reviewed=false`、`not_training_ready=true`。
- **不划分 train/test，不自动训练**；不报告精度、真实地块数或平方米。
- 修改 review 副本后仍需重新质检，导入或另存不等于自动验收。
- 两批来源/空间/时相关系、背景与忽略区规则、跨类优先级仍待人工确认。
- 2026-10-16 Codex 已在非项目 cwd 实际运行转换，得到 **855 candidate / 48 review**（empty 0 / unsupported 0），产物在本机 `reports/labels/RGB-candidates-DATA-20261016-005`；全部仍为待语义复核、非训练就绪（`semantic_reviewed=false`、`not_training_ready=true`）。此为**本机验收批次**，不保证其它资料/环境得到相同数量。
- review 副本面向 Labelme：本轮**未实际打开 Labelme GUI**；仅验证了 JSON 结构、同目录配图、`points` 与副本 PNG 保真，软件兼容性须由实际打开确认。
