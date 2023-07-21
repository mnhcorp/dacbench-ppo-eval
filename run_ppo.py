import argparse
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from utils import make_env, read_config

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.ddqn_local.ddqn import LeadingOnesEval
import shutil

import numpy as np
import pickle
import gzip

class EvalCallback(BaseCallback):
    """
    A custom callback that evaluates the agent every `eval_interval` steps.
    """
    def __init__(self, eval_env, eval_interval: int, n_eval_episodes: int, best_model_save_path: str, result_path: str, **kwargs):
        super(EvalCallback, self).__init__(**kwargs)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path        
        self.result_path = result_path
        self.evaluator = None 

    def _on_step(self) -> bool:
        """
        This method will be called in the model's `learn` method. 
        We evaluate the agent every `eval_interval` steps.
        """
        if self.evaluator is None:
            self.evaluator = LeadingOnesEval(self.eval_env, self.model, agent_type="PPO", log_path=f"{self.best_model_save_path}/eval")
            
        if (self.n_calls + 1) % self.eval_interval == 0:
            self.evaluator.eval(self.n_calls + 1)            

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", "-o", type=str, default="output", help="output folder")
    parser.add_argument("--setting-file", "-s", type=str, help="yml file with all settings")
    parser.add_argument("--num-runs", "-n", type=int, default=1, help="number of training runs")
    args = parser.parse_args()

    # Get configuration from train_conf_ppo.yml
    config_yml_fn = args.setting_file
    exp_params, bench_params, agent_params, train_env_params, eval_env_params = read_config(config_yml_fn)

    if exp_params["n_cores"] > 1:
        print("WARNING: n_cores>1 is not yet supported")

    out_dir = args.out_dir
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
        shutil.copyfile(args.setting_file, os.path.join(out_dir, "config.yml"))

    # Run the training for the specified number of runs
    for run in range(args.num_runs):
        # Create the environment
        env = make_env(bench_params, train_env_params)

        # Write results to a log file
        env = Monitor(env, os.path.join(out_dir, f'run_{run}'))
        
        # Create the evaluation environment
        eval_env = make_env(bench_params, eval_env_params)

        # default values
        batch_size = 2048
        clip_range = 0.2
        gae_lambda = 0.95
        gamma = 0.99
        learning_rate = 0.0003

        if "batch_size" in agent_params:
            batch_size = agent_params["batch_size"]
        if "clip_range" in agent_params:
            clip_range = agent_params["clip_range"]
        if "gae_lambda" in agent_params:
            gae_lambda = agent_params["gae_lambda"]
        if "gamma" in agent_params:
            gamma = agent_params["gamma"]
        if "learning_rate" in agent_params:
            learning_rate = agent_params["learning_rate"]

        # PPO agent
        if agent_params["name"] == "ppo":
            model = PPO('MlpPolicy',
                    env, 
                    verbose=1, 
                    batch_size=batch_size,
                    clip_range=clip_range,
                    gae_lambda=gae_lambda,
                    gamma=gamma,
                    learning_rate=learning_rate)
                    

            # Use the custom callback to evaluate agent's performance after a certain number of steps
            callback = EvalCallback(eval_env, 
                eval_interval=exp_params["eval_interval"], 
                n_eval_episodes=exp_params["eval_n_episodes"], 
                best_model_save_path=os.path.join(out_dir, f'run_{run}'), 
                result_path=os.path.join(out_dir, f'run_{run}', 'eval_infos.gzip'))       
        
            # Train the agent and pass custom callback
            model.learn(total_timesteps=exp_params["n_steps"], callback=callback)

            # Save the model
            model.save(os.path.join(out_dir, f'run_{run}', "ppo_final"))

        else:
            print(f"Sorry, agent {agent_params['name']} is not yet supported")

if __name__ == "__main__":
    main()
