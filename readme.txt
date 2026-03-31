baselines: 对比危险用例搜索方法实现
matlab_scripts: 用以绘图的matlab脚本
pylot/TCP: 其他的端到端或者是模块化自动驾驶实现
rlsan: 强化学习驱动的小生境粒子群搜索
safebench: 原生的safebench代码。learning to collide依赖它
scripts: 基本上分三个函数：run.py(原始safebench库使用)；run_batch_simulation.py(rlsan相关代码使用)
run_rlsan_search.py(使用训练好的强化学习权重,搭配真实物理实验寻找致错样本)


能否在scripts文件夹下面新建一个bash脚本，实现对给定自动驾驶的批量测试。主要任务包括：（1）新建终端开启carla,命令形如：  cd ~/CARLA_0.9.13_safebench/   ，  ./CarlaUE4.sh -carla-port=2028 -RenderOffScreen --no-rendering
（2）进入run_batch_simulation.py，修改'--exp_name'参数，比如为'scenario01_14000'（注意这个数字是递进的，表示当前测试到第14000个用例），修改'--port'参数，这个参数对应（1）中的-carla-port，修改'--tm_port'参数，使得它和'--port'参数一样，修改    test_cases = sampled_parameters[12000:14000,:]中的参数，一般是按照2000递增，（3）修改完毕py文件之后，新建终端执行测试，形如：cd ~/SENSE/，conda activate jzm，python scripts/run_batch_simulation.py

