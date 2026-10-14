# AgroLake 第三方来源与使用状态

登记时间：2026-10-14（Asia/Shanghai）。仅登记本轮实际核查/使用的第三方代码、模型与数据。
核验方式：官方来源网页/API 与本机哈希计算；未确认项保持空白，不臆断。

## 1 代码

| 名称 | 来源 | 版本/提交 | 文件哈希 | 许可 | 允许用途/再分发 | 核实时间 |
|---|---|---|---|---|---|---|
| ftw-tools | PyPI `ftw-tools` | 1.4.3 | wheel SHA256 `33fab944b2394fc1cd9f6c862be8731c14b50fac295669b2b9349c5614275ee1`；sdist SHA256 `3ea58bb311621c272988aff649e7b5e97522690b656cda1f6baec20f4ef12315`（均与 PyPI 官方元数据一致） | PyPI 元数据 license 见包内；代码库 MIT（见 ftw-baselines LICENSE） | 本地运行与评估 | 2026-10-14 |

代码库许可原文（ftw-baselines @ 固定树 `45d7df7ea0c357f2c34182e4a7e0c4625dd43533`）：

```
MIT License

Copyright (c) 2024 kerner-lab
...
```

FTW 的完整运行依赖安装未成功，没有已验收的 `requirements-ftw.lock.txt`，不得将候选版本列作实际依赖。上游来源：[ftw-tools 1.4.3](https://pypi.org/project/ftw-tools/1.4.3/)、[ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines)。本版本网页不导入或执行 FTW。

## 2 模型权重

| 名称 | 来源 | 大小 | 许可/条款 | 结论 |
|---|---|---|---|---|
| `3_Class_CCBY_FTW_Pretrained.ckpt` | [GitHub v1 release](https://github.com/fieldsoftheworld/ftw-baselines/releases/tag/v1) | 204471122 字节，已本地核对 | 上游 README 说明该模型仅用 CC-BY 或同等开放许可资料预训练；仓库代码许可 MIT 与训练数据条款分开记录 | 本地结构验证，尚未完成推理；不在本仓库再分发 |

完整权重本地 SHA256：`d04671c498c181a2bca4b9c1a82fb8dcc93384eaf55841cb6672692f6fb21d3c`。这是本机计算值，不是官方公布摘要。

未使用：`*_FULL_*` 检查点、`2_Class_*` 检查点、任何其他大模型。SAMGeo 未安装。

## 3 数据

| 名称 | 来源 | 大小 / 哈希 | 许可 | 说明 |
|---|---|---|---|---|
| `inference-img.tif` | 固定树 `45d7df7ea0c357f2c34182e4a7e0c4625dd43533` 的 `tests/data-files/inference-img.tif` | 3507684 字节；git blob SHA1 `0c2e1f4ad977684805be126d4d2fb92400e18ec6`（与本机实算一致）；SHA256 `da482e7987b36898bd9f8381e7c7fe95c05dfc5ccbba19a58eb65e544d1af750` | 随 ftw-baselines 仓库（MIT）分发 | 上游测试输入；仅用于输入适配验证，不代表 1.4.3 训练时相定义 |

未下载完整国家数据；未合成 8 波段；未复制 RGB。

## 4 署名与再分发

- 若对外展示，须保留 ftw-baselines（MIT，kerner-lab）与 CC-BY 检查点数据来源署名，并遵守 CC-BY 数据条款。
- 本轮仅本地评估，未公开上传、未再分发、未报名、未提交。

## 5 P0 治理底座依赖（DEV-20261015-001，实际安装并使用）

登记时间：2026-10-15。仅登记本轮 P0 实际安装并使用的直接依赖；FTW 未完成的候选依赖不列入已验证清单。

| 名称 | 来源 | 版本 | 文件哈希（官方 PyPI 元数据） | 许可 | 允许用途/再分发 | 核实时间 |
|---|---|---|---|---|---|---|
| Streamlit | PyPI `streamlit` | 1.50.0（wheel `streamlit-1.50.0-py3-none-any.whl`，10068477 字节，`requires_python !=3.9.7,>=3.9`） | wheel SHA256 `9403b8f94c0a89f80cf679c2fcc803d9a6951e0fba542e7611995de3f67b4bb3` | Apache-2.0 | 本地运行与演示 | 2026-10-15 |
| Pillow | PyPI `pillow` | 11.3.0（wheel `pillow-11.3.0-cp311-cp311-win_amd64.whl`，6986055 字节，`requires_python >=3.9`） | wheel SHA256 `1a992e86b0dd7aeb1f053cd506508c0999d710a8f07b4c791c63843fc6a807ac` | MIT-CMU | 本地运行 | 2026-10-15 |
| 传递依赖 | PyPI | 见 `requirements-p0.lock.txt`（pip freeze 实测） | — | 各自上游许可 | 本地运行 | 2026-10-15 |

说明：上述 wheel 哈希取自官方 PyPI 元数据（安装前核对），本轮使用 `--no-cache-dir` 未在本地重算 wheel 哈希；本机实测安装版本与冻结清单见 `requirements-p0.lock.txt`。SQLite、hashlib、csv、json 为标准库，不引入额外第三方依赖。

标准库与平台：Python 3.11.15（`.venv-p0`）。未安装 GDAL、rasterio、MinIO、MySQL 或地图套件。
