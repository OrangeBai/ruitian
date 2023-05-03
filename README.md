## 使用深度学习模型对时间序列进行预测
### Config
参数设置见`config.yaml`文件，其包含部分设置及多个分段。
#### 
1. `name`: 模型名称
2. `model_dir`: 保存模型的根目录
3. `train`: 模型训练的基本信息
   1. `num_epoch`: number of epoch
   2. `grad_acc`: 由于数据量较大，IO压力较大，因此训练时将batch size设置为1，并使用了梯度累积
4. `model_setting`: 模型设置，包含线性模型、多层DNN、LSTM。
   1. `lienar`: 通过移除DNN的非线性激活函数实现线性回归
   2. `fc_dropout`: 包含dropout的DNN
   3. `LSTM`: 多层LSTM
5. `dataset`: 
   1. `base_dir`: 原始数据目录
   2. `proc_dir`: 预处理目录
   3. `input_UID`: 输入的UID，即所选用进行预测的UID序列，`None`则为全部.
   4. `output_UID`: 目标UID、即目标预测的UID序列， `None`则为全部。
   5. `num_workers`、DataLoader使用的CPU数量。
6. `lt_opt`: linear rate scheduler与optimizer的设置。

### 数据预处理
进行训练前，首先需要对数据进行预处理以提升IO效率并整理数据，该部分由`pre_process.py`实现。

#### 数据检查
对比各个数据集中包含的UID数量是否一致，并对齐UID
#### 保存UID、features
整理并保存UID、features的名称
#### 转存数据
对每个日期下的数据进行整合并写入hdf，并剔除所有`dtype`不为int或float的数据。
#### 计算各个feature的均值方差
首先，便利所有hdf文件，进行数据对齐。首先对所有非0、非NAN的数据feature进行求平均，得到数据集的均值。
随后根据均值计算方差。

### 模型设置
目前分别对比了线性回归、非线性DNN以及LSTM模型。对于loss的计算，为了避免`y=0`时造成的loss过小，
在backward propagation前对所有`target_y=0`的部分赋0。由于内存限制及IO限制，`batch_size`无法调高,
因此使用了梯度累计。
#### 线性回归
深度学习的表达能力主要来来自于激活函数提供的非线性。由于数据量较大难以进行全时序回归，因此采用了全连接层进行拟合。
其中第一层对所有feature进行融合，随后使用所有输入UID对输出UID进行预测，从而达到输入->输出为线性的效果。
#### FC-Dropout
由于feature、UID不具有相邻相关性，因此CNN模型并不适用。作为benchmark，此处采用了DNN + Dropout的结构。
其中模型输入、输出与线性回归一致。
#### LSTM
长短记忆模型可以有效的存储长期、短期的信息，因此在本问题中较为适用。为保持feature的一致性，
输入转置为 `7 * num_feture * num_uid`的结构，首先对UID进行非线性融合，随后以得到的feature作为
LSTM的输入。

### 实验结果
见result.ipynb.
目前LSTM似乎由于参数量过大、学习率过低的问题难以拟合。今天我将重新调一下参数跑一下实验，并将结果更新到github.