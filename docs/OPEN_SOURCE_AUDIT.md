# GitHub 开源替代项目初审

本文件为2026-10-14选型时点的历史记录；候选与当时状态保留。当前实现和已验证版本以 README、模型适配说明及审核后的提交为准。

核对日期：2026-10-14。通过用户指定的 GitHub 插件搜索并读取实际仓库文件。搜索方向包括 Streamlit 地理应用、农业田块边界和数据质量检查。检查范围为 LICENSE、README、依赖声明、目录及部分源码；未安装依赖、下载模型权重或运行推理，因此不把候选写成已在本机跑通。

后续复核：当前首选验证 FTW，暂不并行实施其他模型。本文所记源码 `2.0.0b5` 与 PyPI 发布版 `1.4.3` 是不同版本；后者要求 Python `>=3.11,<3.13`，不得直接套用本表中当前源码的 API/配置。页面底座和模型仍须按具体任务验证后锁定。详细统一口径保留在本地协作资料，未随Git分发。

## 一 建议采用的组合

**页面底座**：采用 [opengeos/streamlit-map-template](https://github.com/opengeos/streamlit-map-template) 的多页面组织和地图展示代码作为可修改参考，独立加入 AgroLake 的接入、索引、质量处置、关联查询和报告模块。参考代码已经缓存，不需要从一个空白页面开始。

**表格治理**：可用 [unionai-oss/pandera](https://github.com/unionai-oss/pandera) 的 DataFrameSchema 和检查机制实现 CSV 类型、枚举与值域校验；依赖未验证前也可用清楚的 Python/Pandas 规则实现同样的接口。它是校验组件，不提供数据湖、版本和业务处置系统。

**AI 应用**：按拿到的数据选择一条路线。满足模型规定的多波段与时间窗口条件时，优先评估 FTW 中许可明确的 CC-BY 检查点；只有普通 RGB 单图时，评估 SAMGeo 的对象候选分割与人工确认流程。两条路线分别验收，不同时安装全部可选引擎，也不把通用对象分割称为五类农业自动识别。

数据质量界面另参考 DataQtor 的“导入、检测、修正、前后报告”流程。其旧依赖和缓存方式需要替换，不建议把整个旧环境作为新项目运行环境。

## 二 候选比较

| 项目 | 主仓库许可与已读文件 | 可复用部分 | 接入门槛或缺口 | 选择结论 |
|---|---|---|---|---|
| [streamlit-map-template](https://github.com/opengeos/streamlit-map-template) | MIT；Home.py、Split Map 页面、requirements.txt、LICENSE | 多页面入口、地图及分屏展示 | 原 requirements 直接包含 GDAL、geopandas 等；没有本项目治理逻辑 | 推荐页面代码底座，按实际功能拆分依赖 |
| [streamlit-geospatial](https://github.com/opengeos/streamlit-geospatial) | MIT；原大纲已核对的地理应用参考 | 多页面和地理展示模式 | 示例范围较广，仍须新增治理模块 | 保留为补充参考 |
| [streamlit-multipage-template](https://github.com/giswqs/streamlit-multipage-template) | 目录和 requirements 已读 | 多页面结构 | 依赖还包括 geemap、GDAL 和 Notebook 代理；不比所选模板更轻 | 本次不作主底座 |
| [dataqtor](https://github.com/baligoyem/dataqtor) | MIT；README、LICENSE、requirements、home.py 前 100 行、utils.py | CSV 质量处置和前后报告流程 | 固定 Streamlit 1.2.0、Pandas 1.2.4、NumPy 1.20.1；源码使用旧 caching 和全局状态，含无关业务规则 | 参考流程，不整套移植旧环境 |
| [pandera](https://github.com/unionai-oss/pandera) | MIT；LICENSE.txt、README 示例 | 表格字段、类型、枚举与值域校验 | 需核实选定版本与 Pandas 兼容性；无图像和业务处置模块 | P0 的可选校验组件 |
| [segment-geospatial](https://github.com/opengeos/segment-geospatial) | MIT；pyproject、官方测试和 README | SamGeo 输入图像、生成掩膜、提示分割、转矢量 | PyTorch、地理依赖及权重；不同模型条款分别核查；测试中的在线地图来源不自动可复用 | 单图辅助提取候选 |
| [ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines) | MIT 代码；pyproject、模型注册表、推理入口 | 真正田块及边界模型、栅格推理、模型输入条件登记 | 当前源码声明 Python >=3.12 且 <3.15，包含较多地理和 ML 依赖；权重条款按注册项区分 | 数据适配时优先评估 CC-BY 检查点 |
| [agribound](https://github.com/montimaj/agribound) | Apache-2.0 主仓库；README 已初审 | 农田边界流程、处理溯源设计 | 多引擎依赖与部分权重另有条款；原搜索结果显示仓库体量较大 | 后续参考，不作为 P0 全量底座 |
| [agricultural-monitoring-ftw](https://github.com/climatechange-ai-tutorials/agricultural-monitoring-ftw) | README 表示 MIT；教程路径可见 | FTW 使用与农田分析教学流程 | 推荐 Colab/GPU；未单独核对完整 Notebook 和其数据条款 | 教程入口，不能据其演示认定本机可用 |
| [field-boundary-delineation](https://github.com/resingm/field-boundary-delineation) | README 声明代码 GPL-3.0、相关数据 CC-BY-4.0 | 田块边界的训练、测试、预测 Notebook | 按 Colab 设计；是研究作业，未核实依赖和数据可获得性 | 本次不选主路线 |

上述选择是基于源码与依赖的初审推断，不是运行基准。未检查的许可细节和实际运行门槛在正式复用前补齐。

## 三 具体从哪里改

| 已核对的上游入口 | AgroLake 的改造位置与任务 | 本项目必须新增的内容 |
|---|---|---|
| 模板 Home.py | 应用首页、数据与任务概览 | 读取真实目录与指标，不展示上游示例结果 |
| 模板 pages/ 及 Split Map 页面 | 数据目录、场景预览、结果对照页面 | 上传与索引绑定、质量状态、适用的本地图层和图像对照 |
| DataQtor home.py 的导入和前后报告流程 | 质量处置页面设计参考 | 遥感和场景规则、隔离、复检、版本、审计记录 |
| Pandera DataFrameSchema 示例 | 属性 CSV 校验适配层 | 农业字段、单位、枚举与语义字典，统一问题报告格式 |
| samgeo 的官方测试调用 | 独立模型适配器：generate 或 set_image/predict，必要时 tiff_to_vector | 输入与权重登记、类别限制、人工确认、结果回写和质量检查 |
| ftw_tools/inference/inference.py 的 run 及 model_registry.py | 独立 FTW 推理适配器及模型选择配置 | 规定输入、窗口和预处理校验；固定配置与设备；运行及输出登记 |

所有模型输出回写统一运行表；与原图、属性、文本复核记录关联。前端、质量规则和模型接口分开，使模型失败不导致数据目录无法启动。

## 四 版本与许可记录

本次查到的上游提交作为定位信息，后续以实际安装和验证的固定版本为准：

| 仓库 | 查到的提交 SHA | 该提交的日期 |
|---|---|---|
| streamlit-map-template | `c79dbd77be65aae4e54fe3d3e8a0488ef64f0473` | 2026-01-20 |
| segment-geospatial | `0d7a4ac1b3f6afe1970ee0ae5c11ba1da6bbb49e` | 2026-09-20 |
| ftw-baselines | `45d7df7ea0c357f2c34182e4a7e0c4625dd43533` | 2026-10-08 |
| pandera | `309c4dc9c7c6f1867d29caf9502da69ce4abff6c` | 2026-10-13 |

FTW [模型注册表](https://github.com/fieldsoftheworld/ftw-baselines/blob/45d7df7ea0c357f2c34182e4a7e0c4625dd43533/ftw_tools/inference/model_registry.py)已分别标注 CC-BY-4.0、混合开放许可和 AGPL 模型。其 CC-BY 田块模型仍有对应时间窗口要求，不能直接用不适配的普通照片输入。正式选用时保存具体权重版本、哈希、条款和署名信息。

MIT/Apache 主仓库代码的声明不代表其所有数据、外部服务和权重都同样许可。复制或改造源码保留相应 LICENSE、作者和修改记录；数据与模型分别登记。AGPL 或自定义条款按选用组件实际核对，不将整个技术栈统一标为 MIT。

## 五 已保存与尚未执行

`work/开源参考_20261014/` 已缓存 13 份只读源码、配置、许可文件或明确标记的摘录，来源与上游文件 SHA 见其中 `来源索引.json`。这是参考资料缓存，不是完整仓库，也不是 AgroLake 可执行项目；部分记录标记为摘录，不能直接当完整模块运行。

下一步在正式开工时核实环境，先锁定页面底座，再验证一套合法小样本与模型。当前未克隆完整仓库、未安装依赖、未下载权重、未运行这些测试，也未建立或修改远程 GitHub 仓库。
