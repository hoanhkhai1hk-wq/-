# RGB 训练前审计与离线可视复核包（DATA-20261016-007；008/009 定向修复）

本模块对 **DATA-20261016-005** 生成的候选/复核批次做**完整训练前审计**，并生成一份
**离线**可视复核包，供人工先看问题、再决定修标。它**只读**输入，不训练、不切分、
不修改任何原始标注/图像/旧产物/服务/网页/模型/配置。

> 本工具只适用于 005 的 `candidate` / `review` 批次。`empty`（无有效像素）与
> `unsupported`（未知 label 或非 polygon）状态**明确不支持并拒绝**：遇到即整体失败、
> 不产生输出目录。若将来需要覆盖这两类，请另建任务，不要改本工具语义。

> DATA-20261016-008 仅做**定向修复**（不重写正确的像素/统计核心）：
> ① 原/副本 JSON 前置解析与结构校验；② policy 校验；③ review_reasons 索引/类型/label；
> ④ 同源与派生产物去重；⑤ 像素统计整数类型；并改进候选左右并排预览与窄屏易读性。

> DATA-20261016-009 **最终小修**（仍不重写已通过的像素/统计核心）：① `parse_tile_name`
> 无 prefix 时返回空串并保留 row/col；② policy 四类身份**严格对应**（0=Field、
> 1=Field ridge、2=Road、255=ignore，允许既有中文说明，拒绝缺类/额外训练类/调换/
> 拼接矛盾描述）；③ 原 JSON `imagePath` 必须**严格等于**原 TIFF basename（不再只比
> basename、不剥目录迁就 `foreign/` 前缀或绝对路径）；④ 候选 mask 与 review 副本必须是
> **真正的 PNG**（`Image.format=='PNG'`），TIFF 改名 `.png` 一律拒绝（原 TIFF 源仍为 RGB，
> 不要求原图为 PNG）；⑤ review 副本只对**原稿**做单向派生（`imagePath`→PNG basename、
> `imageData=null`、仅正向 `Fieldridge`→`Field ridge`）后深比较，**副本侧不归一**，故
> “原 `Field ridge`→副本 `Fieldridge`”逆向改写被拒绝；另把问题 shape 的 `sN` 文字按
> `textbbox` 限制到画布内，避免钉在右/下边缘被截掉（不改源坐标、颜色/alpha/间隔/统计）。

## 输入

- `--input <目录>`：005 候选目录，默认
  `<根>/reports/labels/RGB-candidates-DATA-20261016-005`，必须位于 `reports/labels` 内。
  目录需含 `manifest.json`（含 `records`）、**`policy.json`** 与其中的 `masks/`、`review/`
  派生产物。
- 记录的源文件（`json_path`/`image_path`）必须落在 `<根>/邓`、`<根>/hy0713 (1)`
  或 `reports/labels`（fixture）内；拒绝绝对路径与 `..` 越界。
- 派生路径（`mask_path`、`review_png_path`、`review_json_path`）必须是相对 input 目录、
  且落在 input 目录内。

## 操作（由 Codex 运行；WorkBuddy 只编写）

```powershell
& .venv-p0/Scripts/python.exe -B scripts/labels/build_review_pack.py `
    --input reports/labels/RGB-candidates-DATA-20261016-005 `
    --output reports/labels/DATA-20261016-007-review-pack
```

- 使用现有 `.venv-p0` 解释器与 Pillow；`-B` 不写字节码；**不安装、不联网**。
- 根目录由脚本位置定位，**不依赖 cwd**（可从非项目目录运行）。
- `--output` 只允许 `reports/labels` 内**全新**目录；已存在即拒绝，不覆盖/不删除；
  不得为 `reports/labels` 本身，也不得与输入相同、位于输入下或包含输入。
  省略 `--output` 时自动生成带唯一时间/随机后缀的目录名。

## 审计内容

**所有校验都在创建输出目录之前完成**：任一来源/策略失败即整体失败，末条坏记录也不会
留下半成品输出目录。

1. **逐记录只读校验**：原 JSON/TIFF 的 SHA256/bytes 与 `manifest` 一致、图像为 RGB
   且尺寸一致。
2. **原 JSON 前置解析与结构校验**（不只看 SHA）：根为对象；`imagePath` **必须严格等于**
   原 TIFF basename（字段字符串逐字相等，不剥目录、不含目录或绝对路径）；`imageWidth`/`imageHeight` 为正整数（非 bool）且与记录一致；
   `shapes` 为列表；每个 shape 为对象、`label` 属于已知类别、`shape_type=polygon`、
   `points` 为有限数坐标（非 bool）。**候选不接受空/坏 shapes；review 允许保留已知几何
   问题，但结构必须可渲染，坏 shape 不静默跳过。** 解析/读取错误均包装为 `BuildError`。
3. **候选 mask**：SHA、**真正的 PNG**（`Image.format=='PNG'`）、`L` 模式、尺寸、值域仅 `0/1/2/255`，**实算直方图**并与
   `manifest.pixel_stats` 的 `class_pixels{0,1,2}` / `ignore_pixels` / `denominator`
   逐项核对；候选必须有有效互斥像素。**统计值必须为非负整数（拒绝 float/bool），
   `denominator` 必须为正整数**。候选不得携带 review 理由或 review 副本。
4. **review 副本前置解析与深比较**：副本 PNG/JSON 分别校 SHA，PNG 为**真正的 PNG**、
   RGB 且尺寸一致，并与原 TIFF 逐像素一致；副本 JSON 根为对象、`imagePath` 必须是同目录
   PNG 的纯 basename（不得含目录）、`imageData=null`、宽高一致、`shapes` 数量与结构可渲染；
   以**原稿**为基准单向派生期望视图（`imagePath`→PNG basename、`imageData=null`、
   仅正向 `Fieldridge`→`Field ridge`）后与副本**直接深比较**，**副本侧不做任何归一**，
   其余原字段/points/顺序必须一致。**不得修改原稿/副本，也不从坏副本继续出成功报告。**
5. **policy 校验**：读取并验证可适配 005 真实 `policy.json`——`class_ids` 恰含
   `0/1/2/255`（拒绝缺类/额外训练类），四类身份按类名开头英文 token **严格对应**
   `0=Field`/`1=Field ridge`/`2=Road`/`255=ignore`（允许既有中文说明，拒绝调换/拼接矛盾
   描述，不做随意子串包含）、`no_background_class=true`、
   `alias_assumption.map` 恰为 `{Fieldridge: Field ridge}` 且 `status=pending_semantic_review`、
   `training_policy.semantic_reviewed=false` / `not_training_ready=true` / `no_train_test_split=true`。
   **不硬编码整份 policy 的 SHA，也不忽略真实 005 既有字段**；缺 policy 或坏 JSON 一律拒绝。
6. **review_reasons 校验**：必须为非空列表；每条为对象、`type` 属于
   `canvas_oob`/`self_intersection`/`too_few_distinct_points`、`shape_index` 为 **0 起**
   整数（非 bool）且在范围内、`label` 与对应原 shape 一致。保留全部问题详情与 0 起索引。
7. **同源/派生产物去重**：源 JSON、源 TIFF、候选 mask、review 副本（PNG/JSON）均按
   `resolve + normcase` 去重，同一物理文件不得被多条记录复用。
8. **全量统计**：候选 `class_pixels`/`ignore` 总量、每类有像素的候选图片数、
   各批 candidate/review 数、review 问题类型计数；逐图 CSV 保留来源 SHA/路径/status；
   **review 行的像素统计留空（未生成），不写 0% 已知有效**；汇总仍只计候选。
9. **文件名解析**：可解析 `*_tile_r<row>_c<col>` 时记录 prefix/row/col；
   同时接受 `hy0717_tile_r35_c49` 与 **`tile_r54_c24`（prefix 空但保留 row/col）**。
   这些仅为来源线索，**不把文件夹/前缀当已证地理独立性，不建立任何 train/val/test 划分**。

未知状态/结构、字段不符、`semantic_reviewed!=false` 或 `not_training_ready!=true`
（即被擅自改成训练就绪）一律拒绝。`sample_id` 需为便携安全组件且大小写折叠唯一。

## 可视包内容

- 对**所有** review 记录生成"**左原图 / 右原图+标注轮廓**"预览：按原 JSON 绘制轮廓、
  类别固定颜色、`review_reasons` 的 shape 额外高亮并标 **0 起** `shape_index`；
  越界坐标完整保留于文字。绘制可被画布裁切，但**不修改源坐标**；问题 shape 的 `sN`
  文字按 `textbbox` 将起点限制到画布内，避免贴右/下边缘被截掉（完整问题索引仍在 HTML 文本列出）。
- 从候选中**确定性**选择最多 **12** 张，优先覆盖道路/田埂并跨批次，展示"**左原图 /
  右彩色候选 mask 叠加**"并排预览；`255` 保持原图可见，类 `0` 也显示。不复制全部候选原图。
- `index.html`：UTF-8 静态页、含 `viewport`、`overflow-wrap:anywhere` 与合理容器宽度，
  长路径/问题坐标在窄屏不撑开整页；中文说明、图例、问题列表、样本来源与 PNG/JSON
  副本链接；review/候选图分别注明"左原图/右轮廓（粗线=问题 shape）"与"左原图/右候选掩膜"，
  明确**尚待复核**；`**离线**`/`**不保存**` 等改用 HTML `<strong>`。
  **无 JS 网络请求、无 CDN、无外部服务**；链接相对本产物，经 URL 编码/HTML 转义，
  支持中文/空格，不泄漏绝对本机路径；只显示"待复核"，无按钮假装确认/修标成功。
- `gallery.json`：每个样本的预览路径、来源、状态、问题、引用的派生副本路径（注明
  相对项目根 / 输入目录 / 输出目录基准），供独立验收，不靠解析 HTML 猜对应关系。
- `records.csv`、`audit.json`、`README.md`。

`audit.json` 固定 `training_ready=false` / `semantic_reviewed=false` /
`split_created=false`，并附 `policy_checked` 摘要；待办含：修正实际数量的 review 问题、
全体语义确认、`Fieldridge` 别名待确认、来源空间/时相关系与切分方案待定、
现 8 波段 FTW 不兼容 RGB、无沟渠类。

## 预期结果

- 输出目录内出现 `audit.json`、`records.csv`、`gallery.json`、`index.html`、`README.md`
  与 `previews/`。任一来源/策略校验失败时不产生输出目录。
- 候选数量与 review 数量**按实际**统计，**不硬编码** 903/855/48。

## 下一步

1. 打开 `index.html` 看问题；在标注软件中**另存**修正副本后**重新质检**。
2. 本页**不保存**修正或确认，仅显示"待复核"。
3. 全体语义确认、`Fieldridge` 别名口径、来源空间/时相关系与训练划分仍待人工决定。

## 测试（由 Codex 运行）

```powershell
& .venv-p0/Scripts/python.exe -B -m unittest tests.labels.test_review_pack -v
```

测试只写 `reports/labels/` 下**全新 fixture 目录**（保留不删），覆盖手算像素统计、
255 与类 0、确定性选图、数量不硬编码、末条坏 SHA/坏 JSON 不建输出、原 JSON 结构
（imagePath/宽高/未知 label）、review 副本坏 JSON 与 imagePath/points 改动、policy
缺失/0 改背景/255 改训练类/擅自训练就绪、review_reasons 空/未知类型/索引越界/负数/
非整数/label 不符、同源复用、像素统计 bool/float、错误 mask 直方图、路径越界、
副本像素不符、HTML 转义与空格链接、问题 shape 索引保留、非项目 cwd、候选 255 透明、
左右并排、review CSV 统计留空、重跑拒绝；并补 009 定向案例：无 prefix 保留 row/col、
policy 四类身份调换/拼接/额外类拒绝与有效身份通过、原 `imagePath` 外来/反斜杠/绝对前缀
拒绝、TIFF 伪装 PNG（候选 mask 与 review 副本）拒绝、别名正向通过与逆向/非 label 改动
拒绝、小画布高亮文字不截断且问题索引仍见于 HTML。不触碰真实数据、不安装。

## 边界与未完成

- 候选/副本**不等于**语义已审核；全部 `semantic_reviewed=false`、`not_training_ready=true`。
- **不划分 train/test，不自动训练**；不报告精度、真实地块数或平方米。
- 本轮**未实际打开 Labelme GUI**；仅验证 JSON 结构与副本像素保真，软件兼容性须实际打开确认。
- 两批来源/空间/时相关系、背景与忽略区规则、跨类优先级仍待人工确认；无独立沟渠类别。
- 本工具当前针对 005 批次结构；真实 903 转换的最终审计由 Codex 在审核后运行。
