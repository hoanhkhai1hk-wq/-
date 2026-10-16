# RGB 标注修正接收与复检（LAB-20261016-001）

本模块**独立**接收人工另存的几何修正 JSON，保留原始文件、005 副本与 855 候选，
生成独立复检清单，并对几何通过的修正派生 L 掩膜。它**不自动修 48 份**、
**不伪造人审结论**、**不做真实训练/划分**。当前**没有人工修正**。

工具：`scripts/labels/intake_corrections.py`（只读 005 输入；只写 `reports/labels` 下全新目录）。
WorkBuddy 只编写，由 Codex 运行并独立审核。

## 何时使用

005 候选包把 48 份有几何问题（自交 / 画布越界）的标注单独放进 `review/`，生成可编辑的
`<sample_id>.png` + `<sample_id>.json` 副本。人工在标注工具里**只改 `points`**、**另存**后，
用本工具接收并复检；当前尚无任何另存修正，故全部记为 `pending`。

## 复用与范围

- 复用 `build_review_pack` 的 `_load_manifest` / `_load_policy` / `_validate_record_fields` /
  `_scan_record` 与重复 id/来源检查，对**整个 005 输入**只读审计；复用 `prepare_candidate`
  的点校验/几何/栅格化与 `EPS`。**不复制或重写**既有审计算法（零面积判定为本模块新增）。
- `corrections` 目录必须**已存在**且位于 `reports/labels` 内；不能是 005 输入本身/子目录，
  不能与输出互为祖先；解析真实路径，拒绝逃逸。
- 只接受**平铺** `<sample_id>.json` 子集：拒绝子目录、其它扩展、未知 id、候选 id、
  大小写碰撞、路径逃逸。省略 `--corrections` 即“空提交”（全 `pending`）。
- 输出 `reports/labels` 下**全新**目录，不在输入/修正目录内，也不包含它们；已存在即拒绝。
  省略 `--output` 时生成随机默认目录，但**同样走**输入/修正目录父子范围校验
  （LAB-20261016-004 修补：默认路径不再提前返回绕过校验；输入为 `reports/labels` 本身
  等越界情形会直接拒绝）。
- **所有输入与修正内容完整预检后才创建输出目录**；最后一份坏也不留输出（无半成品）。

## 只允许坐标修正

与 **005 review 原始副本**（`review/<id>.json`，`imagePath=<id>.png`、`imageData=null`）比对：

- 顶层除 `shapes` 外全部**深相等**：`imageData=null`、`imagePath` 保持同目录 PNG basename；
  禁止改图/宽高，禁止加入 `training_ready` 等新字段。
- 每个 shape 的数量/顺序/`label`/`shape_type`/`group_id`/`flags` 及**全部非 points 字段**深相等：
  禁止增删 shape、改类别、改 shape_type。
- 允许每个 shape 的 `points` 改变点数/坐标；坐标必须**有限数且非 bool**。
- 比对为**类型感知深比较**：`bool`/`int`/`float` 分别区分（`8` 改成同值 `8.0` 也会被拒绝）；
  `points` 允许数值等价（只把坐标表示类型 int→float 不算修正，记 `unchanged`）。
- 点数不足、**零面积**（共线）、自交、越界仍记为几何问题（不能凭“仅 3 点互异”把共线视合格）；
  沿用既有 `EPS=1e-6` 口径，不自动裁切/重排/补点，不修改提交原件。

## 结果状态

| 状态 | 含义 |
|---|---|
| `pending` | 未提交修正，不重新猜测坐标 |
| `unchanged` | 已提交，但与 005 副本 `points` 完全一致；**不声称已修** |
| `review` | 已提交但仍存在几何问题（点数不足/零面积/自交/越界） |
| `candidate` | 几何通过且有效**互斥**像素 >0；生成 L 掩膜 |
| `empty` | 几何通过但有效互斥像素为 0 |

- `candidate` 生成 L 模式 PNG（值域 `0/1/2/255`）：`0` 为 Field 田块（**不是背景**），
  `1` 为 Field ridge 田埂（含 `Fieldridge` 待复核拼写归一），`2` 为 Road 道路，
  `255` 为未画/跨类冲突/未确认。
- 所有**有效结构**的提交（含仍 `review`/`unchanged`/`empty`）原字节保存到
  `accepted/<id>.json`，并复制 005 副本配图到 `accepted/<id>.png`，使 `imagePath` 仍可打开；
  未提交项不复制。

## 两目录工作流程（重要）

1. 把 005 目录的 `review/<id>.json` **与**同目录 `review/<id>.png` 复制到一个
   `reports/labels` 下的**另存工作目录**，用标注工具只改 `points`；
2. **只把改好的 `<id>.json` 交回**一个 `reports/labels` 内的**接收目录**
   （PNG 放另一个展示/临时工作区即可，接收目录**不接受** PNG 等其它扩展）；
3. 运行本工具，输出目录 `accepted/` **自带同目录配图** `<id>.png`，`imagePath` 仍能打开。

> 标注工具若改动版本 / flags / 类别 / 宽高等，会被拒绝；请核对另存内容，
> 不能暗中改语义或授权。修改副本后**仍需重新质检**，另存 ≠ 自动验收通过。

## 运行方式（由 Codex 执行）

```powershell
& .venv-p0/Scripts/python.exe -B scripts/labels/intake_corrections.py `
    --input  reports/labels/RGB-candidates-DATA-20261016-005 `
    --output reports/labels/LAB-20261016-002-example
# 有另存修正时再加 --corrections reports/labels/<接收目录>
```

> `--output` 必须是**尚不存在**的全新目录名（示例用 `LAB-20261016-002-example`）；
> 若该目录已存在，请换一个全新名字，工具不会覆盖或删除已存在目录。

> 使用现有 `.venv-p0` 解释器（`-B` 不写字节码）；**不安装、不新增、不改动环境**。
> 默认 `--input` 为 005 候选目录；`--corrections` 省略即空提交；所有相对路径相对项目根，
> 可从**非项目 cwd** 运行。

## 输出目录

| 文件 | 说明 |
|---|---|
| `summary.json` | 计数、来源 manifest/policy SHA、路径基准、状态标志 |
| `records.json` | 逐 review 记录（状态、变化 shape 索引、前后问题、像素、摘要、输出路径与 SHA） |
| `records.csv` | UTF-8 BOM；逐图状态/问题数/像素/提交与输出摘要 |
| `README.md` | 零基础说明 |
| `accepted/<id>.json` | 提交**原字节** |
| `accepted/<id>.png` | 005 副本配图（使 `imagePath` 可打开） |
| `masks/<id>.png` | 几何通过候选的 L 掩膜（0/1/2/255） |

路径基准：`correction.path` 与源路径相对**项目根**；`five_copy.*` 相对 **005 输入目录**；
`outputs.*` 相对**本输出目录**。

## 状态与边界

- `semantic_reviewed=false`、`training_ready=false`、`not_training_ready=true`、
  `split_created=false`；`human_verified` 恒为 false，提交者身份不明。
- `geometry_check_passed` 只是几何复检事实，**不等于**人审语义通过。
- 855 原候选**仅计数**，不合并、不重建整批、不称训练就绪。
- 不训练、不划分 train/val/test、不报精度/真实地块数/平方米；0 不是背景。
- 未新增依赖；不实现模型/API/网页/Labelme 启动；模块 import 不执行、不写目录。

## 测试（由 Codex 运行）

```powershell
& .venv-p0/Scripts/python.exe -B -m unittest tests.labels.test_corrections -v
```

测试只写 `reports/labels/` 下全新 fixture 目录，不使用自动删除的临时目录；
覆盖：空提交 `pending`、`unchanged`（含 `points` 仅 int→float 数值等价）、
蝴蝶形修四边形 → `candidate` 且 16 像素掩膜完整手算、越界仍 `review`、共线零面积 `review`、
跨类覆盖 → `empty`、非有限/bool 拒绝、类别/尺寸/顶层/增删 shape 改动拒绝、
同值 float 宽高与嵌套非 points 数字换类型（`1`→`1.0`、`group_id` `1`→`True`）拒绝、
混入候选 id/未知 id/子目录/其它扩展拒绝、提交文件解析逃逸（mock 链接）拒绝、
末份损坏不留输出、重跑不覆盖、非项目 cwd 相对路径与 UTF-8 BOM；
另有一项**集成回归**：用既有 `prepare_candidate.prepare` 生成 candidate/review 真实 005 风格
输入后再交 `intake`（仅合成 8x8）。

事实（不泛称"本轮未运行"）：002 已由 Codex 实际运行并通过——31 模块 / 23 独立 /
真实 14 核对全过，真实输出 `reports/labels/LAB-20261016-002-intake`；48 份仍 `pending`、
0 提交，没有人工修标、没有真实训练。004（LAB-20261016-004）只补默认输出路径边界：
`_resolve_output(None,...)` 统一走显式路径校验，新增 2 个聚焦回归
（输入为 labels 根时默认输出被拒；正常 fixture 下默认输出可生成全新同级目录）——
Codex 已复验：最终 **33 项模块测试与 23 项独立检查全部通过**；真实 002 批次的 14 项核对与 4 份产物摘要继续有效。004 只改默认目录解析分支，显式路径的全量审计和复检逻辑未变，未重复生成真实批次。

## 未完成

- 当前无人工修正：48 份几何问题**未修**，全部 `pending`；真实训练/划分/评精度/平方米**未做**。
- 提交者身份、48 份语义确认、`Fieldridge` 别名、邓与 `hy0713 (1)` 的来源/时相关系仍待人工答复。
