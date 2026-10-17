# RGB 小群组可拟合性检查

全734张训练的田埂漏分尚未解决。先检查同一张图能否在少量图片共同训练时拟合现有标注，以缩小后续排查范围。工具是`scripts/rgb/cohort_probe.py`，仅执行记忆检查，不发布候选模型。

## 本轮固定条件与结果

使用原有734张实验范围、128×128输入、同一29,779参数SmallUNet、seed20261017、Adam学习率0.001、不增强。权重统一取**全部734张**的平方根逆频，不随小群组改变。每组从相同参数重新初始化，每步看完整群组，固定300次更新，每张恰好被看300次。记录0/50/100/200/300步，以下只取固定最后一步。

表中始终是同一张已见问题图`d6b1d7c1681b10f770d7`（`hy0713 (1)/tile_r68_c43.tif`）的田埂指标，并非群组平均：

|训练内容|田埂precision|田埂recall|田埂IoU|
|---|---:|---:|---:|
|007：只训练该图|87.69%|99.96%|87.65%|
|008 ridge4：该图加3张田埂问题图|80.13%|95.83%|77.43%|
|008 mixed8：上述4图再加4图|99.76%|50.46%|50.40%|

群组整体田埂IoU另为ridge4 **51.63%**、mixed8 **25.66%**；它们覆盖不同图片，不与表中单图数字混作同一统计量。

加入其他图片后，这张图在固定训练预算下更难拟合。群组组成、一次更新的平均目标和计算量也随之改变；这些结果不能证明标签冲突、容量不足或某个优化因素是唯一原因。4/8图检查也不是原734图训练过程的缩小复现。旧004模型和默认训练策略保留，007/008模型均不晋升为候选。

这是与现有训练标注的一致性，不是独立农业准确率。来源、时间、Fieldridge含义仍未知；48份排除、121份含未确认别名的整图暂缓继续有效，没有独立验证/测试划分。

## 如何运行

先准备一个新配置文件，放在`reports/rgb/`下，输出目录必须与配置和选集所在目录互不包含。配置示意：

```json
{
  "schema_version": 1,
  "selection_sha256": "当前selection.json文件的SHA256",
  "cohorts": [
    {"name": "example2", "sample_ids": ["允许实验训练的ID1", "允许实验训练的ID2"], "reason": "选择依据"}
  ]
}
```

实际008配置在`reports/rgb/Codex-cohort-20261017-008/cohorts.json`。以下使用它运行一次新检查；`manual-cohort-01`必须尚不存在：

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B 'scripts/rgb/cohort_probe.py' --selection 'reports/rgb/TRN-20261017-003-selection/selection.json' --cohorts 'reports/rgb/Codex-cohort-20261017-008/cohorts.json' --output 'reports/rgb/manual-cohort-01' --steps 300 --size 128 --seed 20261017 --threads 4
```

只使用已有recognition环境，不下载模型。默认300步，允许1—500步；尺寸32—256且为8的倍数；线程1—8。最多8组，每组1—16个不重复ID，组名必须安全且忽略大小写后唯一。所有ID须在选集中标为`experiment_train`；不能通过配置重新纳入排除/暂缓图片。

工具先重新核对全部原图/原JSON、manifest、policy、选集决定与mask摘要，再开始训练。每组独立优化器、相同初始化；检查实际更新次数、曝光次数、loss/梯度/参数有限及参数改变，保存后严格重载核对。选集或配置运行中发生改变时不能发布成功摘要。

## 看哪些产物

- 根`summary.json`：两个群组的来源绑定和最后整体指标。
- 每组`summary.json`：所有里程碑、逐图/全组混淆矩阵、权重、更新与曝光次数、参数和文件摘要。
- `final_per_image.csv`：每图固定最后一步田埂指标。缺失分母留空，JSON中为null；不补0或1。
- 每图PNG与`-comparison.png`：128×128预测及原RGB/现有标注/预测三联图。预览是诊断像素最近邻放大2倍，不是512原分辨率推理。绿色田块、黄色田埂、红色道路，标注灰色为忽略。
- `probe_checkpoint.pt`：专用`agrolake_rgb_cohort_probe_v1`，标为`not_candidate_model=true`。正常候选预测和全量诊断入口都会拒绝，避免误作正式模型使用。

混淆矩阵真实为行、预测为列；0是田块，255不计分母。中间记录保留完整，只有最后模型另存checkpoint，不能声称已重新推理验证每个中间模型。

实际输出：`reports/rgb/TRN-20261017-008-cohort-probes/`。独立复核从原RGB、原mask和两个checkpoint重新推理全部12个末态预测，逐像素复算矩阵并检查三联图、CSV和摘要；证据在`reports/rgb/Codex-cohort-20261017-008/independent-actual.json`。

## 图块一致性只读检查

009仅在734张原图的64×64、步长64网格上寻找**RGB完全相同**的跨图纹理块，纹理条件为最大单通道空间标准差大于5。有命中时才对双方均非255的mask像素比较类别；相同RGB也不能判定哪份标注正确。

实测46,976块中46,846块满足纹理条件，130块跳过，跨图完全相同命中0。独立复核换用BLAKE2b并保存每个摘要的全部出现位置，同样无跨图命中。结果在`reports/rgb/TRN-20261017-009-patch-audit/`。

这是网格对齐、完全相同图块的有限搜索，不能排除任意偏移、缩放、光照变化或近似重复，也不能证明标签一致、来源独立或同一时相。因此本轮仍不建立独立数据划分或改标注。

## 软件验证

```powershell
& '.\.venv-recognition\Scripts\python.exe' -B -m unittest tests.recognition.test_rgb_fitting tests.recognition.test_rgb_diagnostics tests.recognition.test_rgb_cohort_probe -v
```

测试使用合成小图检查曝光/初始化/全池权重、保存加载和拒绝边界；不把软件测试通过当成模型效果通过。原图、原JSON、旧模型、FTW网页和环境保持原样，DeepSeek继续暂缓。
