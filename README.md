## 特定进球风格的足球AI模仿学习（基于GRF 5v5环境）
### **Tip:**
在项目目录下执行以下命令，创建专家数据集文件夹，将准备好的hdf5数据集放置在`.grf/datasets`中
```bash
mkdir -p .grf/datasets
```
### **Example:**
执行如下代码即可进行训练（或者查看`grf_imitation/scripts/run_gail.py`文件，自己撰写bash脚本训练）
```bash
python examples/football_gail.py
or
./examples/football_gail.sh
```

执行如下代码可对某个训练好的Model进行评估
```bash
python examples/football_evaluate.py
```
### **Framework:**
![framework](figure/framework.png)

### **Hyper-Parameters:**
|参数名|作用|备注|
| -- | -- | -- |
|expert|专家数据集的名称||
|env-name|环境名称|5v5环境的名字为"malib-5vs5"|
|n-itr|一次训练的Iteration次数|n-itr * step-per-itr即为本次训练的总时间步数|
|step-per-itr|一个Iteration的时间步数|即一个Iteration采样长度为step-per-itr * 8(因为有8个球员)的Batch Sample|
|num-workers|采样Worker的数目|注意Worker的数目尽量和step-per-itr的数目具有整数倍关系，避免不必要的轨迹截断|
|tabular-log-freq|实验log输出的频率(每tabular-log-fre个Iteration输出一次)||
|param-log-freq|NN参数保存的频率(每param-log-freq个Iteration保存一次)||
|video-log-freq|Render视频保存的频率(每video-log-freq个Iteration保存一次)||
|disc-lr|GAIL discriminator的学习率|一般要小于Policy的学习率|
|disc-update-num|GAIL discriminator的一个Iteration下更新的次数|一般较小，我们不希望Disc太强|
|layers|GAIL Policy Net, Disc Net的隐藏层大小|一般限制在64-256之间，层数不要太深|
|activation|GAIL 激活函数|64用tanh，大于64用ReLU|
|lr-schedule|GAIL 学习率衰减|实例：'Pi: [[0, 1.0], [10000, 0.1]];Disc: [[0, 1.0], [10000, 0.1]]', 即Policy学习率在10000个Iteration后衰减为1/10, Disc的学习率同理|
|buffer-size|PPO Buffer的大小|Buffer的大小为step-per-itr的8倍(因为有8个球员的数据)|
|repeat-per-itr|PPO 一个Batch重复更新的次数|一般设为10-20|
|batch-size|PPO minibatch的大小|一般为Buffer Size的1/10|
|entropy-coeff|PPO 熵激励奖励的系数||
|ret-norm|PPO 累计奖励的归一化||
|adv-norm|PPO 优势函数的归一化||
