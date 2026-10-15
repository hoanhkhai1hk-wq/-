# RGB 标注准备

本模块为第一批真实 RGB 切片做**人工查看/标注的输入准备**（任务 DATA-20261015-003）。
它只产出一份“待标注包”，**不含任何标签**：所有样品 `annotation_status = unlabeled`、
`label_path = null`。空标注既不是负样本，也不代表已标注。

- 脚本：`scripts/data/prepare_annotation_pack.py`
- 测试：`tests/data/test_annotation_pack.py`
- 依赖：仅标准库与 Pillow（复用 `.venv-p0`，不新增依赖）

## 为什么需要它

- 380 张 RGB 切片是 3 通道、**没有配套标注**；**时相身份未核实**，不能充当 8 波段双时相输入；现有 FTW 权重需要 8 波段双时相输入，二者**不兼容**（禁止复制通道凑 8 通道）。
- 目标类别尚未确定。先把“值得人工看/标的少量代表切片”整理成一份可复现、可核对的包，让标注工作能立即开始，而不依赖尚未回复的类别定义。

## 它做什么

1. 读取可信索引 `reports/data/DATA-20261015-002/manifest.json`（JSON 列表，`path` 相对原图根）。
2. 校验：路径必须相对、仅两个已知批次、符合 `tiles/tile_rN_cN.tif`；SHA256 为 64 位小写十六进制、`bytes` 为正整数；无重复路径/坐标；解析后仍在原图根内（拒绝绝对路径、`..` 与 symlink 逃逸）；索引中的 SHA/字节与实际文件逐一对照，任一不符即**非零退出**。
3. 选图（每批固定 6 张，组内不足则全取）：先取最接近行列包围盒中心的一张，再逐次取“到已选集最小距离最大”的未选图；距离用归一化 `(row, col)` 上的平方欧氏距离，**用 `Fraction` 精确表达**（不依赖浮点 `==`），中心取包围盒真正中点（归一化后恒为 `1/2`）。同分按 `numeric(row, col, path)` 升序。行/列跨度为 0 时按 1 处理。所有分支（含空组、不足 6 张）统一返回 `path → 1-based 排名` 的 `dict`。
4. 导出**逐像素一致**的 512×512 RGB PNG（不拉伸、不增强、不重新标色）；若原切片不是 RGB 或尺寸不符，直接报错，不静默转换放行。
5. 生成联系图、待标注清单和零基础说明。

选图只是按行列的**空间覆盖建议**，不宣称视觉或统计代表性，也不代表真实地块分布。

## 用法

在项目根目录（`D:\AI_Workspace\Tasks\兴智杯`）执行：

```powershell
& .\.venv-p0\Scripts\python.exe -B scripts/data/prepare_annotation_pack.py
```

默认：源根 `data/raw/user_uploads/20261015`，索引 `reports/data/DATA-20261015-002/manifest.json`，
输出自动生成到 `reports/data/annotation-pack-<时间>-<随机>`。

常用参数：

| 参数 | 说明 |
|---|---|
| `--source-root` | 原图根目录；**只能**是既有归档根 `data/raw/user_uploads/20261015`，或项目 `reports/data` 内的 fixture 根（拒绝其它来源，避免读到别的项目/凭据） |
| `--manifest` | 可信索引 `manifest.json`；解析后**必须**位于项目 `reports/data` 内 |
| `--output` | 输出目录；**必须**位于项目 `reports/data` 内且**不存在**（拒绝覆盖/逃逸） |
| `--per-group` | 每批挑选数量，默认 6 |
| `--tile-size` | 切片边长（用于校验），默认 512 |

- 相对参数一律**锚定项目根**（不依赖 Host 默认 cwd）；绝对参数原样使用。
- 输入范围（`source_root`/`manifest`）与输出约束由 CLI 和公开 `build_annotation_pack` 入口**统一校验**，直接调用 build 也不能绕过。
- 从任意工作目录都可运行（默认路径相对脚本 `__file__` 定位项目根），`--help` 正常输出。

## 输出内容

```
reports/data/annotation-pack-<suffix>/
├─ png/                      逐像素一致的原尺寸 RGB PNG
├─ contact_sheet.png         带编号（AP-01…）与批次/行列的联系图
├─ manifest.json             来源/SHA/字节/批次行列/PNG 路径尺寸/待标注状态
├─ annotation_todo.csv       UTF-8 BOM，可直接用 Excel 打开
└─ README.md                 零基础操作步骤
```

`manifest.json` 关键字段：`schema_version`、`source_manifest_sha256`、`source_root`、
`selection_policy`、`total_input`、每批 `input_count`/`selected_count`、`target_status=pending`、
`target_classes=null`、来源信息（meta 旧路径，标注“待核实”），以及每个样品的
`source_relative_path`/`source_sha256`/`source_bytes`/`group`/`row`/`col`、
`png_relative_path`/`png_sha256`/`png_size`、`annotation_status=unlabeled`、`label_path=null`。
**不含任何分割掩膜。**

## 测试

```powershell
& .\.venv-p0\Scripts\python.exe -B tests/data/test_annotation_pack.py -v
```

覆盖：选图确定性与中心/远端、组不足取全部（经 `select_items` 端到端）、正好 6 张、空组、精确同分
用 `Fraction` 命中 numeric 同分（r1c1 而非 r1c2）、PNG 与 TIFF 逐像素一致、路径越界与坏 SHA/字节拒绝、
被篡改原图拒绝、已有输出不覆盖、原图不变、未标注语义、输入/输出范围（build 入口亦校验）、CLI 非零退出边界。
测试只在 `reports/data/DATA-20261015-004/` 下写全新自测子目录，不触碰真实原图与旧索引。

## 边界与未决

- 本模块**不训练、不推理、不猜测类别**，也不做 train/test 划分。
- 两批次是否同一源地块未知，不能当作同一地块的时相配对被直接使用。
- 原图无地理参考，面积只能按像素计，不换算平方米。
- 目标类别、代表性与划分方案仍待用户确定；来源（拍摄方式/日期/正射处理/作物身份）待核实。
