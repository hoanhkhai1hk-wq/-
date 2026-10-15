# AgroLake 第三方来源与使用状态

登记时间：2026-10-14（Asia/Shanghai），运行状态更新至2026-10-15。仅登记实际核查/使用的第三方代码、模型与数据。
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
| `3_Class_CCBY_FTW_Pretrained.ckpt` | [GitHub v1 release](https://github.com/fieldsoftheworld/ftw-baselines/releases/tag/v1) | 204471122 字节，已本地核对 | 上游 README 说明该模型仅用 CC-BY 或同等开放许可资料预训练；仓库代码许可 MIT 与训练数据条款分开记录 | 已通过直接SMP适配器和本地HTTP服务完成真实CPU推理，634模型张量严格加载；不在本仓库再分发 |

完整权重本地 SHA256：`d04671c498c181a2bca4b9c1a82fb8dcc93384eaf55841cb6672692f6fb21d3c`。这是本机计算值，不是官方公布摘要。

未使用：`*_FULL_*` 检查点、`2_Class_*` 检查点、任何其他大模型。SAMGeo 未安装。

## 3 数据

| 名称 | 来源 | 大小 / 哈希 | 许可 | 说明 |
|---|---|---|---|---|
| `inference-img.tif` | 固定树 `45d7df7ea0c357f2c34182e4a7e0c4625dd43533` 的 `tests/data-files/inference-img.tif` | 3507684 字节；git blob SHA1 `0c2e1f4ad977684805be126d4d2fb92400e18ec6`（与本机实算一致）；SHA256 `da482e7987b36898bd9f8381e7c7fe95c05dfc5ccbba19a58eb65e544d1af750` | 随 ftw-baselines 仓库（MIT）分发 | 上游测试输入；仅用于输入适配验证，不代表 1.4.3 训练时相定义 |

未下载完整国家数据；未合成 8 波段；未复制 RGB。

运行身份限制：目前只对上列公开适配样例完成真实模型演示；时相与波段顺序尚未独立核实，没有真值精度评估。服务软件测试另使用明确标注SYNTHETIC的fixture，不能把这些占位二进制或合成规则测试当作真实模型结果。

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

## 6 直接SMP适配器新装小包（DEV-20261015-006，实际安装并使用）

登记时间：2026-10-15。`.venv-recognition`（Python 3.14.7，`--system-site-packages` 只读继承 base CPU Torch）中以 `pip --no-deps` 实际安装的 6 个包。wheel 哈希为本机对实际下载 wheel 重算的 SHA256，并与官方 PyPI JSON 元数据 digest 逐项比对一致；证据 wheel 保留于 `.venv-recognition/wheel-evidence/`。

| 名称 | 来源 | 版本 | 本地重算 wheel SHA256 | 许可（dist-info 实读） | 用途 |
|---|---|---|---|---|---|
| segmentation-models-pytorch | [PyPI 0.5.0](https://pypi.org/project/segmentation-models-pytorch/0.5.0/) | 0.5.0 | `c34e09047771aa4dd8878b4f899e8125700cd1f8f7db16e58c37204154151a05` | 主要为 MIT，部分文件另有许可；保留上游 LICENSES 与文件声明 | 构建与严格加载 Unet/efficientnet-b3 |
| timm | PyPI | 1.0.15 | `5a3dc460c24e322ecc7fd1f3e3eb112423ddee320cb059cc1956fbc9731748ef` | Apache-2.0 | SMP encoder 依赖 |
| huggingface-hub | PyPI | 0.27.1 | `1c5155ca7d60b60c2e2fc38cbb3ffb7f7c3adf48f824015b219af9061771daec` | Apache-2.0 | timm/SMP 依赖（未联网下载模型） |
| safetensors | PyPI | 0.6.2 | `c7b214870df923cbc1593c3faee16bec59ea462758699bd3fee399d00aac072c` | Apache-2.0 | SMP 依赖 |
| PyYAML | PyPI | 6.0.3 | `4a2e8cebe2ff6ab7d1050ecd59c25d4c8bd7e6f400f5f82b96557ac0abafd0ac` | MIT | SMP 依赖 |
| tifffile | PyPI | 2025.6.11 | `32effb78b10b3a283eb92d4ebf844ae7e93e151458b0412f38518b4e6d2d7542` | BSD-3-Clause | 替代 rasterio/GDAL 读取 GeoTIFF |

继承只读复用（未重装/未升级）：numpy 2.5.2、Pillow 12.3.0、tqdm、requests、filelock、fsspec、packaging、typing_extensions、torch 2.14.1+cpu、torchvision 0.29.1+cpu。未安装 GDAL、rasterio、torchgeo、Lightning 或其他大包；本轮未重新下载模型权重。FTW v1 CC-BY 检查点与上游署名要求见上文第 2/4 节，继续保持。
