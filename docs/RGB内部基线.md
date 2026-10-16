# RGB内部训练与预测

2026-10-17。这是独立CPU命令行模块，使用自己的三通道RGB资料训练一个小型U-Net。与既有FTW双时相8波段模型、识别服务和网页分开，未修改它们的输入协议或权重。

## 当前真实结果

- 903份带标注资料中，用户决定排除48份几何问题标注，保留原件，不再等待修正。
- 855份候选中，121份包含未确认含义的Fieldridge（无空格），整份暂缓；734份原标签写法明确，可用于内部拟合。
- 用户回答Fieldridge含义与两批资料来源关系“不清楚”，因此保持unknown，未创建宣称独立的验证集、测试集。
- 734份已完成3轮、276次Adam参数更新，模型保存和严格重载通过；生成了训练样本上的原图/标注/预测对照。
- 新实验许可不改旧候选的`not_training_ready=true`，所有语义复核仍为false。这不是正式泛化评估，没有独立准确率、真实地块数或平方米结果。

3轮预览发现田埂样本漏分显著，因此另存相同配置、相同seed、相同数据的10轮对照（920次更新），两个模型均保留。10轮道路样本区域更完整，但田埂样本仍明显漏分、边界不稳，**当前识别效果未验收为可用**。训练loss下降不替代这项判断；原标签语义、来源独立性与模型效果仍待进一步核实。

类别：0田块、1田埂、2道路、255忽略。0不是背景；没有沟渠标签。未画区与跨类别冲突继续为255；不把田埂当作沟渠。

## 环境与路径

使用现有`.venv-recognition/Scripts/python.exe`（Torch CPU、Numpy、Pillow），不需安装或联网。所有相对路径均相对项目根`D:/AI_Workspace/Tasks/兴智杯`，与Shell当前目录无关。每个输出必须是`reports/rgb`下全新目录；存在即拒绝，失败产物保留。输入与输出目录不能互为父子。

以下PowerShell示例使用全新`manual-*`目录，第一次执行后再次执行会拒绝覆盖。请更换输出名，不要改旧记录。

## 第一步：产生选择清单

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/real_baseline.py' select --input 'reports/labels/RGB-candidates-DATA-20261016-005' --output 'reports/rgb/manual-selection-01'
```

输入是既有候选包，完整核对原图、原JSON与掩膜摘要/像素分布后，输出`selection.json`、UTF-8 BOM的`records.csv`、`summary.json`、`README.md`。按原JSON标签判定，不信任单独的别名标记：

|disposition|含义|
|---|---|
|excluded_user_decision|原review，用户排除|
|deferred_alias|原JSON含Fieldridge，整份暂缓|
|experiment_train|其余candidate，只许可内部实验|

全部903条仍在清单里，便于追踪被排除和暂缓的来源。来源摘要、重复引用、原标签映射、别名记录或像素统计不符时，拒绝且不创建输出。

## 第二步：训练内部基线

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/fit_baseline.py' train --selection 'reports/rgb/manual-selection-01/selection.json' --output 'reports/rgb/manual-fit-01' --epochs 3 --size 128 --batch-size 8 --seed 20261017 --threads 4
```

再次从原JSON重建全部去向并核对来源、选择清单、原mask，防止改清单把排除项混入训练。只缓存experiment_train。RGB双线性缩放，mask最近邻缩放保留255。模型是随机初始化的两层encoder-decoder U-Net，通道8/16/32、跳跃连接、3入3出，共29779参数；不是最终参赛网络。

Adam学习率0.001；使用忽略255的加权交叉熵。类别权重为`sqrt(全部有效训练像素/(3*该类像素))`，再除以权重均值。每轮损失用加权损失之和除以目标类别权重之和，避免直接平均大小不同的batch；同时保存每轮所有sample_id和batch大小。每轮必须覆盖全部样本一次，最后不足整批也训练。

每步检查logits、loss、梯度、更新后参数是否有限，异常不写成功摘要。输出`checkpoint.pt`、`summary.json`、`history.csv`、`README.md`，包含数据/选集摘要、seed、版本、配置、参数摘要和优化步数。固定参考batch在更新前后分别计算loss，同一口径，不能解释成真实精度。保存后用`weights_only=True`和strict重载核对同输入logits。

参数范围：epochs 1..20，size 32..512且为8的倍数，batch-size 1..32，threads 1..8，seed 0..2^32-1；API中布尔值不当整数接受。先从默认值开始，大尺寸/大批量将增加CPU内存和时间。

## 第三步：查看训练样本预测

从选择清单的experiment_train记录取sample_id。以下示例实际数据中的道路样本也属于该范围：

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/fit_baseline.py' predict --checkpoint 'reports/rgb/manual-fit-01/checkpoint.pt' --selection 'reports/rgb/manual-selection-01/selection.json' --sample-id '02180e6a041796b0af2d' --output 'reports/rgb/manual-predict-01'
```

输出`prediction.png`（原尺寸、L模式、0/1/2）、`overlay.png`、`comparison.png`（左原图、中标注、右预测叠加）、`pixels.csv`和`summary.json`。绿色田块、黄色田埂、红色道路，标注中的灰色是忽略区。模型logits双线性上采样回原尺寸后取argmax，预测整图三类，不把原标注255伪装成预测类别。

像素表是模型的整图像素计数，不是田块数量、精度或实际面积。这个命令只演示本模型的训练样本；它不是任意图片上传服务，也没有接入网页。

## 已保存的本轮产物

- `reports/rgb/TRN-20261017-003-selection/selection.json`：实际训练绑定的选择快照。
- `reports/rgb/TRN-20261017-003-fit/`：真实734份、3轮内部拟合与模型。
- `reports/rgb/TRN-20261017-003-predict-road/`：道路训练样本对照。
- `reports/rgb/TRN-20261017-003-predict-ridge/`：田埂训练样本对照。
- `reports/rgb/TRN-20261017-003-predict-field/`：纯田块训练样本对照。
- `reports/rgb/TRN-20261017-004-fit/`：相同条件10轮对照模型。
- `reports/rgb/TRN-20261017-004-predict-road/`、`...-predict-ridge/`、`...-predict-field/`：10轮同样本预测，田埂漏分仍保留。
- `reports/rgb/TRN-20261017-003-reviewed-selection/`：接续审查后的选择模块重建快照，用于与实际训练快照逐ID核对；不覆盖训练绑定的原快照，不重训或修改checkpoint。

训练快照由Codex初版选择实现生成，随后统一保留WorkBuddy选择模块并补严格字段检查。初版源码在`reports/rgb/Codex-real-20261017-001/source-snapshots/`留存，选集摘要与模型绑定不改。两个选择快照的元数据格式可不同，实际样本去向与来源应一致。

## 验证与下一步

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B -m unittest tests.recognition.test_rgb_fitting -v
```

测试只生成保留的合成fixture；真实训练证据单独存放。后续需取得来源关系和独立标注依据，设计可靠的空间/时相划分，再做模型比较和效果评估。RGB网页服务接入、未见样本推理入口及DeepSeek尚待后续；此轮不能据拟合loss宣称农业效果已验收。
