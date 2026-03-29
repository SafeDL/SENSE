"""
用于大规模获取test matrix的仿真测试数据
1、例如grid search获取一批初始数据,用于真值对比
2、需要注意的是,CARLA仿真测试的随机性,可能会导致一些随机碰撞,必要的时候需要做二次验证
"""
import os.path as osp
import os
import torch
import pickle as pkl
import argparse
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner_simple import CarlaRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='scenario05_10000')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--auto_ego', type=bool, default=False)
    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    # 对于相同测试route的场景只能开启多个server的方式并行
    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2020, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8020, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    args = parser.parse_args()
    args_dict = vars(args)

    # 解析和加载配置文件(主要是被测试的自动驾驶模型)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # load agent config
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    # load scenario config
    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)

    with open(os.path.join('/home/hp/SENSE/log', 'sampled_parameters.pkl'), 'rb') as f:
        sampled_parameters = pkl.load(f)
    test_cases = sampled_parameters[8000:10000,:]
    runner = CarlaRunner(agent_config, scenario_config, step_by_step=False)
    runner.run(test_cases)

