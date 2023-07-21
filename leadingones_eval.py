import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

import sys
curDir = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(curDir)
scriptDir = os.path.dirname(curDir)
sys.path.append(scriptDir)

from dacbench.envs.policies.theory.calculate_optimal_policy.run import calculate_optimal_policy
from runtime_calculation import expected_run_time, variance_in_run_time

class LeadingOnesEval():
    def __init__(
        self,
        eval_env,
        agent, 
        use_formula: bool=True, # use runtime_calculation script instead of running eval_env
        n_eval_episodes_per_instance: int = 5,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = False,
        verbose: int = 1,
        name: str = "",
        agent_type: str = "DDQN",   # default to DDQN agent
    ):
        self.eval_env = eval_env
        self.verbose = verbose
        self.agent = agent
        self.name = name
        self.log_path = log_path
        self.agent_type = agent_type

        # check if agent_type is valid
        assert self.agent_type in ["DDQN", "PPO"], f"ERROR: agent_type must be either 'DDQN' or 'PPO', but got {self.agent_type}"

        if save_agent_at_every_eval:
            assert log_path is not None, "ERROR: log_path must be specified when save_agent_at_every_eval=True"

        # create log_path folder if it doesn't exist
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        
        # Detailed logs will be written in <log_path>/evaluations.npz
        self.detailed_log_path = None
        if log_path is not None:
            self.detailed_log_path = os.path.join(self.log_path, "evaluations")
 
        self.use_formula = use_formula
        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval
        
        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = eval_env.instance_id_list

        # best/last mean_runtime of each instance
        self.best_mean_runtime = [np.inf] * len(self.inst_ids)
        self.last_mean_runtime = [np.inf] * len(self.inst_ids)

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = [] 
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []

        # element i^th: 
        #   - policy at self.eval_timesteps[i]
        #   - its runtime per instance (sorted by self.inst_ids)
        #   - a list of number of decisions made per episode for each instance (for TempoRL)
        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        self.eval_n_decisions = []
        
        if hasattr(eval_env, "action_choices"):
            self.action_choices = eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        if self.verbose>=1:
            print("Optimal policies:")
            
        for inst_id in self.inst_ids:

            inst = self.instance_set[inst_id]
            n = inst["size"]

            # get the optimal policy
            if self.discrete_portfolio:
                portfolio = [k for k in sorted(eval_env.action_choices[inst_id], reverse=True) if k<n]
                _, policy = calculate_optimal_policy(n, portfolio)
            else:
                policy = [int(n/(i+1)) for i in range(n)]
            self.optimal_policies.append(policy)
            
            # calculate the runtime of the optimal policy
            runtime_mean = expected_run_time(policy, n)
            runtime_std = np.sqrt(variance_in_run_time(policy, n))
            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

            if self.verbose>=1:
                print(f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}")
            if self.verbose>=2:
                print("\t" + " ".join([str(v) for v in policy]))   


    def eval(self, n_steps, with_skip=False) -> bool:
        if self.verbose>=1:
            print(f"steps: {n_steps}")
        
        self.eval_timesteps.append(n_steps)
        
        policies = []
        policies_unclipped = []
        runtime_means = []
        runtime_stds = []
        n_decisions = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            if self.agent_type == "DDQN":
                # get current policy on this instance
                #policy_unclipped = [self.agent.act(np.asarray([n,fx]), 0) #TODO: only works for observation space [n,fx]
                #                        for fx in range(n)]
                policy_unclipped = self.agent.get_actions_for_all_states(n)
                if self.discrete_portfolio:
                    policy_unclipped = [self.action_choices[inst_id][v] for v in policy_unclipped]
                policy = [np.clip(v,1,n) for v in policy_unclipped]
            else:
                obs = [[n, i] for i in range(n)]
                actions, _ = self.agent.policy.predict(obs)
                if self.discrete_portfolio:
                    policy_unclipped = [self.action_choices[inst_id][a] for a in actions]
                policy = [np.clip(v,1,n) for v in policy_unclipped]

            policies.append(policy)
            policies_unclipped.append(policy_unclipped)

            # calculate runtime of current policy
            if self.use_formula:
                runtime_mean = expected_run_time(policy, n)
                runtime_std = np.sqrt(variance_in_run_time(policy, n))
            else:
                # set self.eval_env's instance_set to a single instance (inst_id)
                self.eval_env.instance_id_list = [inst_id]
                self.eval_env.instance_index = 0
                self.eval_env.instance_set = {inst_id: inst}
                # evaluate on the current instance (inst_id)
                episode_rewards = []
                episode_n_decisions = []
                for ep_id in range(self.n_eval_episodes_per_instance):
                    s, _ = self.eval_env.reset()
                    ep_r = 0 # episode's total reward
                    ep_d = 0 # number of decisions made by this episode
                    d = False
                    while True:
                        a = self.agent.act(x=s, epsilon=0)
                        skip = 1
                        if with_skip:
                            skip_state = np.hstack([s, [a]])  # concatenate action to the state
                            skip = self.agent.get_skip(skip_state, 0) + 1
                        for _ in range(skip):
                            ns, r, tr, d, _ = self.eval_env.step(a)
                            ep_r += r
                            if d:
                                break
                        ep_d += 1
                        if d:
                            break
                    episode_rewards.append(ep_r)
                    episode_n_decisions.append(ep_d)
                # set self.eval_env's instance_set back to its original values
                self.eval_env.instance_id_list = self.inst_ids
                self.eval_env.instance_set = self.instance_set
                # calculate runtime mean/std
                runtime_mean = np.abs(np.mean(episode_rewards))
                runtime_std = np.abs(np.std(episode_rewards))
                # save n_decisions info
                n_decisions.append(episode_n_decisions)
                
            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)
            
            if self.verbose>=1:
                print(f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}")
            if self.verbose>=2:
                print("\t" + " ".join([str(v) for v in policy]))

        if self.detailed_log_path is not None:                
            # save eval statistics
            self.eval_policies.append(policies)
            self.eval_runtime_means.append(runtime_means)
            self.eval_runtime_stds.append(runtime_stds)
            self.eval_n_decisions.append(n_decisions)

            np.savez(self.detailed_log_path,
                    inst_ids=self.inst_ids,
                    optimal_policies=np.array(self.optimal_policies, dtype=object),
                    optimal_runtime_means=self.optimal_runtime_means,
                    optimal_runtime_stds=self.optimal_runtime_stds,
                    eval_timesteps=self.eval_timesteps,
                    eval_policies=np.array(self.eval_policies, dtype=object),
                    eval_policies_unclipped=np.array(self.eval_policies_unclipped, dtype=object),
                    eval_runtime_means=self.eval_runtime_means,
                    eval_runtime_stds=self.eval_runtime_stds,
                    n_decisions=self.eval_n_decisions,
                    instance_set=self.instance_set
                )
            # save current model
            if self.save_agent_at_every_eval:
                #self.agent.save_model(f"{os.path.dirname(self.log_path)}/model_{n_steps}")
                if self.agent_type == "DDQN":
                    self.agent.save_model(os.path.join(self.log_path, f"model_{n_steps}"))
                else:
                    self.agent.save(os.path.join(self.log_path, f"model_{n_steps}"))
                

        # update best_mean_runtime
        if self.log_path:
            self.last_mean_runtime = runtime_means
            # how many instances where we get infs
            n_best_infs = sum([v==np.inf for v in self.best_mean_runtime])
            n_cur_infs = sum([v==np.inf for v in runtime_means])
            # mean runtime across instances, inf excluded
            best_overall_mean = np.ma.masked_invalid(self.best_mean_runtime).mean()
            cur_overall_mean = np.ma.masked_invalid(runtime_means).mean()
            # update best
            if (n_cur_infs < n_best_infs) or ((n_cur_infs==n_best_infs) and (cur_overall_mean < best_overall_mean)):
                self.best_mean_runtime = runtime_means
                if self.verbose >= 1:
                    print(f"\t[env: {self.name}] New best mean runtime! ({runtime_means})")
                if self.log_path:
                    if self.agent_type == "DDQN":
                        self.agent.save_model(os.path.join(self.log_path, "best_model"))
                    else:
                        self.agent.save(os.path.join(self.log_path, "best_model"))
                    
                    #print(DQN.load(os.path.join(os.path.join(self.log_path, "best_model"))) #DEBUG
        
        return runtime_means, runtime_stds
        
