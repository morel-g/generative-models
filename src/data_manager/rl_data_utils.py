import os
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import warnings
import numpy as np
import torch
import random
from torch.utils.data import Dataset, random_split

from src.case import Case
from src.data_manager.dataset import Dataset


# missing_packages = []

# try:
#     import gym
# except ImportError:
#     missing_packages.append("gym")

# try:


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


#     with suppress_output():
#         # d4rl prints out a variety of warnings
#         import d4rl

# except ImportError:
#     missing_packages.append("d4rl")

# if missing_packages:
#     warning_message = "Warning: The following packages could not be imported: {}. RL features may not work.".format(
#         ", ".join(missing_packages)
#     )
#     warnings.warn(warning_message, ImportWarning)


class RLDataUtils:
    mean_states = None
    std_states = None
    env_name = None
    env = None
    state_dim = None
    action_dim = None
    horizon = None

    @staticmethod
    def normalize(x):
        states = RLDataUtils.get_states(x)
        mean = RLDataUtils.get_states(RLDataUtils.mean_states)
        std = RLDataUtils.get_states(RLDataUtils.std_states)
        normalized_states = (states - mean.to(states.device)) / std.to(states.device)
        x_normalized = RLDataUtils.replace_states(x.clone(), normalized_states)

        return x_normalized

    @staticmethod
    def denormalize(x):
        states = RLDataUtils.get_states(x)

        mean = RLDataUtils.get_states(RLDataUtils.mean_states)
        std = RLDataUtils.get_states(RLDataUtils.std_states)
        denormalized_states = states * std.to(states.device) + mean.to(states.device)

        x_denormalized = RLDataUtils.replace_states(x.clone(), denormalized_states)

        return x_denormalized

    @staticmethod
    def get_env(env_name):
        import gym

        env = gym.make(env_name)
        RLDataUtils.env_name = env_name
        RLDataUtils.env = env
        RLDataUtils.state_dim = env.observation_space.shape[0]
        RLDataUtils.action_dim = env.action_space.shape[0]

        return env

    @staticmethod
    def prepare_rl_dataset(name=Case.hopper_medium_v2, horizon=100, jump=0):
        with suppress_output():
            # d4rl prints out a variety of warnings
            import d4rl
        env = RLDataUtils.get_env(name)
        data = env.get_dataset()

        dataset = RLDataUtils.create_trajectories(data, horizon, jump)
        traj_dataset = torch.tensor(RLDataUtils.cat_states_actions(dataset))
        RLDataUtils.horizon = traj_dataset.shape[1]
        # traj_dataset = traj_dataset[:256]

        train_size = int(0.9 * len(traj_dataset))
        test_size = len(traj_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(traj_dataset), generator=generator)
        train_indices = indices[:train_size]
        test_indices = indices[train_size : train_size + test_size]
        train_dataset = traj_dataset[train_indices]
        test_dataset = traj_dataset[test_indices]

        RLDataUtils.mean_states = train_dataset.mean(dim=0)
        RLDataUtils.std_states = train_dataset.std(dim=0)

        train_dataset = RLDataUtils.normalize(train_dataset)
        test_dataset = RLDataUtils.normalize(test_dataset)

        return Dataset(train_dataset), Dataset(test_dataset)

    @staticmethod
    def create_trajectories(data, horizon=32, jump=0):
        env_name = RLDataUtils.env_name
        if "hopper" in env_name:
            return RLDataUtils.create_hopper_trajecories(data, horizon, jump)
        elif "maze2d" in env_name:
            return RLDataUtils.create_maze2d_trajectories(data, horizon)
        else:
            raise ValueError(f"Can't create trajectories for env {env_name}")

    @staticmethod
    def create_hopper_trajecories(data, horizon, jump=0):
        data["end_flags"] = [
            t or ti for t, ti in zip(data["terminals"], data["timeouts"])
        ]
        states = data["observations"]
        actions = data["actions"]
        end_flags = data["end_flags"]
        rewards = data["rewards"]

        actions_dataset = []
        states_dataset = []
        rewards_dataset = []
        start_id = 0
        while start_id <= actions.shape[0] - horizon:
            if (
                not np.any(end_flags[start_id : start_id + horizon - 1])
                # or end_flags[start_id + horizon - 1]
            ):
                actions_dataset.append(actions[start_id : start_id + horizon])
                states_dataset.append(states[start_id : start_id + horizon])
                rewards_dataset.append(rewards[start_id : start_id + horizon])

                # Move to the next start_id with jump
                start_id = (
                    start_id + horizon
                    if end_flags[start_id + horizon - 1]
                    else start_id + 1 + jump
                )
            else:
                # Find the next start_id where end_flag is True
                next_true_flag = np.argmax(end_flags[start_id : start_id + horizon])
                start_id += next_true_flag + 1

        dataset = {
            "actions": actions_dataset,
            "states": states_dataset,
            "rewards": rewards_dataset,
        }

        return dataset

    @staticmethod
    def create_maze2d_trajectories(data, horizon):
        data["end_flags"] = [
            t or ti for t, ti in zip(data["terminals"], data["timeouts"])
        ]
        states = data["observations"]
        actions = data["actions"]
        end_flags = data["end_flags"]
        rewards = data["rewards"]

        def group_simulations(end_flags, obs, actions, rewards):
            end_flags[-1] = True
            # Convert to NumPy arrays for efficient computation
            end_flags_np = np.array(end_flags)
            obs_np = np.array(obs)
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)

            # Find indices where new simulations start
            split_indices = np.where(end_flags_np)[0] + 1

            # Split the arrays based on these indices
            grouped_obs = np.split(obs_np, split_indices)[:-1]
            grouped_actions = np.split(actions_np, split_indices)[:-1]
            grouped_rewards = np.split(rewards_np, split_indices)[:-1]

            return grouped_obs, grouped_actions, grouped_rewards

        states_dataset, actions_dataset, rewards_dataset = group_simulations(
            end_flags, states, actions, rewards
        )

        states_dataset = [d[: min(horizon, len(d)), ...] for d in states_dataset]
        actions_dataset = [d[: min(horizon, len(d)), ...] for d in actions_dataset]
        rewards_dataset = [d[: min(horizon, len(d)), ...] for d in rewards_dataset]

        dataset = {
            "states": states_dataset,
            "actions": actions_dataset,
            "rewards": rewards_dataset,
        }

        return dataset

    @staticmethod
    def pad_states_actions(states, actions):
        # Check if both lists have the same number of arrays
        if len(states) != len(actions):
            raise ValueError("Both lists must have the same number of arrays.")

        # Determine the maximum number of rows for each pair of arrays
        max_n = max(s.shape[0] for s in states)

        # Pad each pair of arrays
        padded_states = []
        padded_actions = []

        def pad_array_with_last_element(array, max_n):
            n, d = array.shape
            if n < max_n:
                padded_array = np.empty((max_n, d), dtype=array.dtype)
                padded_array[:n] = array
                padded_array[n:] = array[-1]
            else:
                padded_array = array
            return padded_array

        def pad_array_with_zeros(array, max_n):
            n, d = array.shape
            if n < max_n:
                padded_array = np.zeros((max_n, d), dtype=array.dtype)
                padded_array[:n] = array
            else:
                padded_array = array
            return padded_array

        for s, a in zip(states, actions):
            padded_states.append(pad_array_with_last_element(s, max_n))
            padded_actions.append(pad_array_with_zeros(a, max_n))

        return padded_states, padded_actions

    @staticmethod
    def cat_states_actions(dataset):
        env_name = RLDataUtils.env_name
        if "maze2d" in env_name:
            padded_states, padded_actions = RLDataUtils.pad_states_actions(
                dataset["states"], dataset["actions"]
            )
            padded_states = np.stack(padded_states, axis=0)
            padded_actions = np.stack(padded_actions, axis=0)

            return np.concatenate([padded_states, padded_actions], axis=-1)
        if "hopper" in env_name:
            return np.concatenate([dataset["states"], dataset["actions"]], axis=-1)

    @staticmethod
    def replace_states(x, normalized_states):
        state_dim = RLDataUtils.state_dim
        x[..., :state_dim] = normalized_states
        return x

    @staticmethod
    def get_actions(x):
        state_dim = RLDataUtils.state_dim
        return x[..., state_dim:]

    @staticmethod
    def get_states(x):
        state_dim = RLDataUtils.state_dim
        return x[..., :state_dim]

    @staticmethod
    def get_total_dim():
        return RLDataUtils.state_dim + RLDataUtils.action_dim

    @staticmethod
    def create_state_mask(nb_samples, nb_start_frame=1, nb_end_frame=0):
        state_dim = RLDataUtils.state_dim
        shape = (
            nb_samples,
            RLDataUtils.horizon,
            RLDataUtils.state_dim + RLDataUtils.action_dim,
        )
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[:, :nb_start_frame, :state_dim] = True
        if nb_end_frame != 0:
            mask[:, -nb_end_frame:, :state_dim] = True
        return mask

    @staticmethod
    def create_random_state(id):
        x = torch.tensor(RLDataUtils.env.reset(), dtype=torch.float)
        # torch.tensor(
        #     random.choice(RLDataUtils.env.empty_and_goal_locations), dtype=torch.float
        # )
        # torch.nn.functional.pad(x, (0, RLDataUtils.state_dim - x.shape[0]))
        return (
            x - RLDataUtils.get_states(RLDataUtils.mean_states[id])
        ) / RLDataUtils.get_states(RLDataUtils.std_states[id])

    @staticmethod
    def create_random_states(nb_samples, id):
        cond_list = []
        for _ in range(nb_samples):
            state = RLDataUtils.create_random_state(id)
            cond_list.append(state)
        x_cond = torch.stack(cond_list, dim=0)
        return x_cond
