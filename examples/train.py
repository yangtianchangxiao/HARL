"""Train an algorithm."""
import argparse
import json
import sys
import os
import yaml
import pickle

sys.path.append('/home/cx/HARL')
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

import objgraph

# objgraph.show_growth(limit=10)  # 限制输出的对象类型数量

def main():
    
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="EnvDrone4",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "EnvDrone4"
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="Envdrone4",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    

    if args["load_config"] == "Envdrone4":  # load config from corresponding yaml file
        base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        algo_cfg_path = os.path.join(base_path, "harl", "configs", "algos_cfgs", "hasac.yaml")
        with open(algo_cfg_path, "r", encoding="utf-8") as file:
            algo_args = yaml.load(file, Loader=yaml.FullLoader)
            env_args = {
            "state_type": "FP",
            "another_key": "another_value",
            # ... 更多的键值对
            }

    elif args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    # update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]


    # env
    train_path = os.path.join('/home/cx/happo', 'light_mappo/envs', 'resize_scale_120', 'train_data.pickle')
    # test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
    with open(train_path, 'rb') as tp:
        data = pickle.load(tp)
    
    # record the index of map where all the targets are found during training
    with open ("/home/cx/happo/envs/EnvDrone/classic_control/map_index.txt","w") as w:
        w.truncate(0)

    
    map_num = len(data)

    # start training
    from harl.runners import RUNNER_REGISTRY
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args, mapset = data, map_num = map_num)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
