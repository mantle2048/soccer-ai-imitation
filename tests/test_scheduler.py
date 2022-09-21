# +
import numpy as np
import argparse
from grf_imitation.infrastructure.utils.lr_scheduler import LinearSchedule, PiecewiseSchedule, MultipleLRSchedulers
from grf_imitation.infrastructure.utils.pytorch_util import build_mlp
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# -

itr = 2000
liner = LinearSchedule(itr, initial_p=3e-4, final_p=1e-5)
x = range(itr*2)
val = []
for t in x:
    val.append(liner.value(t))
fig, ax = plt.subplots(1,1)
ax.plot(x, val)

itr = 1000
piece = PiecewiseSchedule(endpoints=[[0, 1e-4], [itr // 2, 1e-5], [itr, 1e-3]])
x = range(itr*2)
val = []
for t in x:
    val.append(piece(t))
fig, ax = plt.subplots(1,1)
ax.plot(x, val)

net = build_mlp(10, 2, layers=[64, 64])
optim1 = optim.Adam(net.parameters(), lr=1e-3)
optim2 = optim.Adam(net.parameters(), lr=1e-3)
optim3 = optim.Adam(net.parameters(), lr=1e-3)
schedule1 = PiecewiseSchedule(endpoints=[[0, 1.0], [100, 0.1]])
schedule2 = PiecewiseSchedule(endpoints=[[0, 1.0], [100, 0.01]])
schedule3 = PiecewiseSchedule(endpoints=[[0, 1.0], [100, 0.001]])
lr_schedule1 = LambdaLR(optim1, lr_lambda = schedule1)
lr_schedule2 = LambdaLR(optim2, lr_lambda = schedule2)
lr_schedule3 = LambdaLR(optim3, lr_lambda = schedule3)
lr_schedules = MultipleLRSchedulers(a=lr_schedule1, b=lr_schedule2)
lr_schedules.update(c=lr_schedule3)
print(lr_schedules)

for _ in range(200):
    optim1.step()
    optim2.step()
    optim3.step()
    print(lr_schedules)
    lr_schedules.step()


