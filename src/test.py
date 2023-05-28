import os
from environment.custom_env import CCEnv
from helper.context import src_dir, entry_dir
from stable_baselines3 import PPO
from helper.utils import get_private_ip
from helper.utils import parse_training_config
from argparse import ArgumentParser
from starter import Starter
from stable_baselines3 import PPO
from datetime import datetime
import json
import numpy as np
import pickle

model_name = "ppo"
params = parse_training_config()
model_dir = os.path.join(entry_dir, "log", model_name, "model")
config_dir = os.path.join(entry_dir, "log", model_name, "config")
obs_dir = os.path.join(entry_dir, "log", model_name, "obs")

def test(args):
    starter = Starter(trace=args.trace, iperf_dir="log/ppo/iperf", iperf_time=args.test_time, ip=get_private_ip())
    env = CCEnv(
        num_features = params["num_features"],
        window_len = params["window_len"], 
        num_fields_kernel = params["num_fields_kernel"], 
        jiffies_per_state = params["jiffies_per_state"],
        num_actions = starter.nchoices, 
        steps_per_episode = params["n_steps"], 
        step_wait_seconds = params["step_wait_seconds"], 
        comm = starter.netlink_communicator, 
        moderator = starter.moderator, 
        reward_name = "Orca"
    )
    try:
        starter.start_communication(tag=f"{args.trace}")
        obs = env.reset()
        # Load the model
        model = PPO(policy = "MlpPolicy", env = env, verbose=1, n_epochs=params["n_epochs"], n_steps=params["n_steps"])
        model.load(os.path.join(model_dir, args.model_name))
        # Test the model
        start = datetime.now()
        features = []
        while not starter.moderator.is_stopped():
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            features.append(info["obs"])
            # env.render()
            if done:
                obs = env.reset()
                break
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        starter.stop_communication()
        starter.close_kernel_channel()
        os._exit(1)
    # Save kernel features
    filename = f"{args.trace}.{starter.get_timestamp()}.npy"
    with open(os.path.join(obs_dir, filename), "wb") as f:
        pickle.dump(features, f)
    # Save test log
    filename = os.path.join(config_dir, f"{model_name}.json")
    with open(filename, "r") as file:
        config = json.load(file)
    models = config.get("models", [])
    for model in models:
        if model["model_name"] == args.model_name:
            if not "tests" in model.keys():
                model["tests"] = []
            # runs = model.get("runs", [])
            info = {
                "trace": args.trace,
                "timestamp": starter.get_timestamp(),
                "test_time": str(datetime.now()-start)
            }
            model["tests"].append(info)
            break
    with open(filename, "w+") as file:
        json.dump(config, file, indent=4)        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--trace", type=str)
    parser.add_argument("--test_time", type=int, default=60)
    args = parser.parse_args()
    test(args)
    os._exit(0)