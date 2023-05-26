"""
TODO: refactor
"""

# Input parameters
import os
import sys
from environment.custom_env import CCEnv
from helper.context import src_dir, entry_dir
from helper.utils import parse_training_config
from starter import Starter
from stable_baselines3 import PPO
from environment.custom_env import CCEnv
from helper.utils import get_private_ip
from helper.utils import time_to_str
from datetime import datetime
import json
from argparse import ArgumentParser

# Inputs
model_name = "ppo"
params = parse_training_config()
model_dir = os.path.join(entry_dir, "log", model_name, "model")
config_dir = os.path.join(entry_dir, "log", model_name, "config")
json_filename = f"{model_name}"
json_filepath = os.path.join(entry_dir, "log", "ppo", "config", json_filename)

def create_config(model_name):
    filename = os.path.join(entry_dir, "log", model_name, f"{model_name}.json")
    if not os.path.isfile(filename):
        data = {
            "models": [],
            "tests": []
        }
        with open(os.path.join(entry_dir, filename), "w") as file:
            json.dump(data, file, indent=4)

def train(args):
    trace = args.trace
    starter = Starter(trace=trace, iperf_dir="log/ppo/iperf", iperf_time=86400, ip=get_private_ip())
    env = CCEnv (
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
    starter.start_communication(tag=f"{trace}")
    model = PPO(policy = "MlpPolicy", env = env, verbose=1, n_epochs=params["n_epochs"], n_steps=params["n_steps"])
    filename = os.path.join(entry_dir, "log", model_name, "config", f"{model_name}")
    if not os.path.isfile(filename):
            create_config(filename)
    if not(args.retrain):
        try:
            model_ts = time_to_str()
            start = datetime.now()
            model.learn(total_timesteps=params["n_steps"])
            if not(params["debug"]):
                model.save(os.path.join(entry_dir, "log", "ppo", "model", f"{model_name}.{model_ts}"))
                # Save the training parameters to a JSON file
                filename = os.path.join(config_dir, f"{model_name}.json")
                with open(filename, "r") as file:
                        config = json.load(file)
                info = {
                    "model_name": f"{model_name}.{model_ts}",
                    "timestamp": model_ts,
                    "training_params": params,
                    "runs": [{
                        "trace": trace,
                        "timestamp": starter.get_timestamp(),
                        "training_time": str(datetime.now()-start)
                    }]
                }
                config["models"].append(info)
                with open(filename, "w+") as f:
                    json.dump(config, f, indent=4, sort_keys=True)
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            starter.stop_communication()
            starter.close_kernel_channel()
            os._exit(1)
    else:
        model_files = os.listdir(model_dir)
        filename = os.path.join(config_dir, f"{model_name}.json")
        with open(filename, "r") as f:
            config = json.load(f)
        if model_files:
            # Sort the model files based on their modification time
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            most_recent_model: str = model_files[0]
            model_path = os.path.join(model_dir, most_recent_model)
            model.load(model_path)  # Load the most recent model
            print(f"Loaded model: {most_recent_model}")
            start = datetime.now()
            try:
                model.learn(total_timesteps=params["n_steps"])
                if not(params["debug"]):
                    model.save(model_path) # overwrite the model with the new one
                    models = config.get("models", [])
                    for model in models:
                        if model["model_name"] == most_recent_model:
                            # runs = model.get("runs", [])
                            info = {
                                "trace": trace,
                                "timestamp": starter.get_timestamp(),
                                "training_time": str(datetime.now()-start)
                            }
                            model["runs"].append(info)
                            break
                    with open(filename, "w") as file:
                        json.dump(config, file, indent=4)
            except Exception as e:
                print(f"An error occurred during training: {str(e)}")
                starter.stop_communication()
                starter.close_kernel_channel()
                os._exit(1)
    env.close()
    starter.stop_communication()
    starter.close_kernel_channel()
            
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--trace", type=str)
    args = parser.parse_args()
    train(args)
    os._exit(0)

# FOR TESTING
# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()