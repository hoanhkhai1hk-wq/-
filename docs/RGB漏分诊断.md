# RGB训练集漏分诊断与权重对照

本模块对`fit_baseline.py`产生的内部模型执行**全部训练样本诊断**，明确不作为独立测试或田间泛化评估。48份已排除、121份别名待定继续暂缓；原标注、旧模型与旧产物不改。

## 为什么补这个步骤

上轮10轮模型在3张预览中有田埂漏分。完整734张、128×128诊断确认：有效标注像素中田块93.61%、田埂4.26%、道路2.13%。模型整体训练像素正确率94.10%，但“全部判成田块”也有93.61%；因此不能只报整体数字。

旧模型的训练集田埂precision约31.01%、recall约31.30%、IoU约18.45%。实际预测田埂总数505650，接近标注501075，但位置正确的只有156823。**预测数量接近不表示识别位置正确。** 两批资料都存在漏分，类别不平衡是已确认的现象，不是唯一原因的因果证明。

## 诊断命令

使用现有`.venv-recognition`，无安装、无联网。所有相对路径相对项目根，输出必须为`reports/rgb`下全新子目录。以下命令第一次运行后再次运行会拒绝覆盖，请另起输出名。

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/diagnose.py' --checkpoint 'reports/rgb/TRN-20261017-004-fit/checkpoint.pt' --selection 'reports/rgb/TRN-20261017-003-selection/selection.json' --output 'reports/rgb/manual-diagnostic-01' --batch-size 8 --threads 4
```

程序重核checkpoint与选择快照绑定、903条去向、全部源摘要/掩膜，预测734个experiment_train样本各一次；不做优化、反向传播或自动重训。`--threads`范围1..8，`--batch-size`1..32。诊断使用该checkpoint训练时的分辨率；RGB双线性、原mask最近邻，忽略255。

输出：

- `summary.json`：全量3×3混淆矩阵、逐类指标、majority常量基线、来源/模型/分辨率/完整ID。
- `records.json`：逐图矩阵、指标、来源及预测PNG摘要，可逐项求和回全量。
- `records.csv`：逐图三类precision/recall/IoU，UTF-8 BOM；无分母的指标留空。
- `predictions/<sample_id>.png`：诊断分辨率的L索引图，0/1/2。这些不是原尺寸展示图。
- `README.md`：范围与口径。

## 指标口径

矩阵**真实类别为行，预测类别为列**。0田块、1田埂、2道路；0不是背景，255完全排除分母。每类TP为对角，FN为行和减TP，FP为列和减TP：

- precision=TP/(TP+FP)：模型判为该类的像素中，有多少与现有标注位置一致。
- recall=TP/(TP+FN)：该类现有标注像素中，模型覆盖多少。
- IoU=TP/(TP+FP+FN)：交集除并集。
- macro IoU按并集非零的类别平均；分母零的单类指标用JSON null，不补0或1。

所有分数都只是对当前训练标注的一致性诊断。现有语义未经独立人工复核、来源关系unknown、没有独立验证/测试划分，不应作为正式农业准确率。majority基线固定输出训练集中像素最多的类别，不使用模型。

## 类别权重单因素对照

`fit_baseline.py train`新增可选`--class-weighting`，默认`sqrt_inverse`保持原行为。`inverse_frequency`改用完整逆频。

|选项|未归一权重|归一方式|
|---|---|---|
|sqrt_inverse|sqrt(全部有效像素/(3×该类像素))|除以三类权重均值|
|inverse_frequency|全部有效像素/(3×该类像素)|除以三类权重均值|

旧策略按全局像素量乘权重计算，田块占73.30%的静态加权像素总量。完整逆频使三类的这个总量相等；但优化器每个batch仍按本batch目标权重和归一，所以不能据此保证梯度贡献或效果相等。

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/fit_baseline.py' train --selection 'reports/rgb/TRN-20261017-003-selection/selection.json' --output 'reports/rgb/manual-inverse-fit-01' --epochs 10 --size 128 --batch-size 8 --seed 20261017 --threads 4 --class-weighting inverse_frequency
```

再用相同诊断命令，把checkpoint换成新模型、output换成新目录。比较时固定数据、seed、模型、轮数、分辨率和指标口径，同时看三类precision/recall/IoU及误分代价。**两个策略的训练loss权重不同，不直接用loss大小评优。** 完整逆频可能提高召回，也可能增加田埂误报，不自动替代原默认或通过效果验收。

## 批次分母对照

完整逆频005实测没有改善田埂：训练集recall从31.30%降到3.26%、IoU从18.45%降到3.04%，不替代旧模型。总体训练像素正确率却上升到95.55%，再次说明只看总体数字会掩盖少数类恶化。

005的920批中，原分母相对“有效像素数×全局期望权重”的比例为0.356到2.926。这是常见每批加权均值的正常性质，不等于原代码计算错误，也不证明是田埂漏分根因。因此006再保持005所有条件，仅改变损失分母做一次对照。

`--loss-normalization batch_weight_sum`仍为默认；`global_expected_weight`把分母改为本批有效像素数乘全训练有效像素的期望权重。全局期望为`sum(各类像素数×各类权重)/全部有效像素数`，两种方式都忽略255。

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/fit_baseline.py' train --selection 'reports/rgb/TRN-20261017-003-selection/selection.json' --output 'reports/rgb/manual-global-fit-01' --epochs 10 --size 128 --batch-size 8 --seed 20261017 --threads 4 --class-weighting inverse_frequency --loss-normalization global_expected_weight
```

新分母不依赖本批类别组成，按有效像素数加权组合子批次时，loss及梯度与一次整批计算一致（已用独立性质测试核对）。这只是优化目标的候选选择，不保证效果提升；原参考batch loss仍使用相同的加权CE均值，便于同权重条件下核对。

所有策略都不根据预测自动修改原标签。语义、来源和独立测试条件未明时，不把“训练得更贴合标注”当作“标注已正确”或“田间已可用”。

## 本轮实测结论（仅训练集，128×128）

|模型/策略|田埂precision|田埂recall|田埂IoU|三类macro IoU|
|---|---:|---:|---:|---:|
|旧10轮，平方根逆频/每批分母|31.01%|31.30%|18.45%|68.18%|
|005，完整逆频/每批分母|31.19%|3.26%|3.04%|63.44%|
|006，完整逆频/全局期望分母|21.65%|6.48%|5.25%|64.40%|

两个新变体都未优于旧模型，不改变默认策略，不自动替换模型。这里只比较与同一734张现有训练标注的一致性，不能据此选择正式参赛模型或声称泛化精度。类别权重与分母不是本轮得到验证的解决方案。

另对固定问题图做300步单图记忆检查：同一小U-Net和128输入能达到该图田埂IoU87.65%。这不是新候选模型，不是独立效果；只说明该设置能拟合这一个样本，不能确定多图失败的唯一根因。专用probe checkpoint使用不同schema，已有预测入口会拒绝把它当正常候选模型加载。

六份定向待复核样例在`reports/rgb/TRN-20261017-005-hard-cases/README.md`，包含漏分和误报示例。它们都来自734张，未重新纳入48排除或121暂缓。按诊断挑选的难例不是随机抽样，不证明原标注错误；原图与原JSON不改。

后续008已完成4图/8图各300步记忆检查：同一问题图田埂IoU分别77.43%/50.40%，多图拟合仍困难，不能认定唯一根因。009固定网格图块检查未找到跨图完全相同纹理块，不能证明独立来源或标签一致。完整条件、逐图与整体指标区别见[RGB小群组检查](RGB小群组检查.md)。

## 本轮产物位置

- `reports/rgb/TRN-20261017-005-baseline-diagnostic/`：旧10轮模型全734张训练诊断。
- `reports/rgb/TRN-20261017-005-inverse-fit/`：仅改变权重的10轮新模型与训练记录。
- `reports/rgb/TRN-20261017-005-inverse-diagnostic/`：新模型同口径诊断，实际运行状态以摘要为准。
- `reports/rgb/TRN-20261017-006-global-fit/`与`...-global-diagnostic/`：仅改变分母的模型和同口径诊断。
- `reports/rgb/TRN-20261017-005-hard-cases/`：六份对照现有标注的待复核样例。
- `reports/rgb/TRN-20261017-007-overfit-probe/`：仅单图300步记忆检查，不能作为候选模型。
- `reports/rgb/Codex-diagnosis-20261017-005/`：独立逐像素复算、受控配置核对、保护基线与审核证据。

测试命令：

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B -m unittest tests.recognition.test_rgb_fitting tests.recognition.test_rgb_diagnostics -v
```

测试使用手算矩阵、忽略值/空分母、源篡改、合成小样本集成与禁止训练守卫。真实734张诊断另外从原图、原mask、checkpoint全量复算，不把软件测试或训练集指标当独立效果证据。
