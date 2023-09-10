import sys
import os

sys.path.append(os.getcwd())

from src.case import Case
from main import parser_to_data
import itertools
import subprocess
from src.eval.plots_2d import write_to_file
from src.data_manager.data import Data
from src.training.training_module import run_sim
from utils import write_to_file
import json

LOGGER_PATH = "outputs/comparison_2d"
SUMMARY_FILE = os.path.join(LOGGER_PATH, "comparison_summary.txt")
PARAMS = {
    "nb_samples": 1e5,
    "data_type": Case.moons,
    "model_type": Case.score_model,
    "epochs": 100,
    "batch_size": 512,
    "T_final": 1.0,
    "nb_time_steps": 1000,
    "nb_time_steps_eval": 1000,
    "beta_case": Case.constant,
    "score_type": Case.singular_score,
    "adapt_dt": not False,
    "gradient": False,
    "gamma": 0.5,
    "decay_case": Case.sig_sig,
    "enable_progress_bar": False,
    "check_val_every_n_epochs": 5,
    "print_ot_costs": True,
    "limit_num_threads": False,
    "accelerator": "cpu",
    "logger_path": LOGGER_PATH,
}
CUSTOM_PARAMS = {
    # "adapt_dt": [True, False],
    "gamma": [0.25, 0.5, 1.0],
    # "score_function_case": [Case.classic_score, Case.time_exp],
}
# Adapt the parameters from PARAMS depending on the values of CUSTOM_PARAMS
ADAPT_PARAMS = {
    "gamma": {
        0.25: {"T_final": 4 * 3.0},
        0.5: {"T_final": 2 * 3.0},
        1.0: {"T_final": 3.0},
    }
}


# def nested_update(d, u):
#     for k, v in u.items():
#         if isinstance(v, dict):
#             d[k] = nested_update(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d


def adapt_params_to_custom(params_dict):
    for k in ADAPT_PARAMS.keys():
        params_dict.update(ADAPT_PARAMS[k][params_dict[k]])
    # params_dict
    # if params_dict["beta_case"] == Case.constant:
    #     params_dict["T_final"] = 3.0
    # if params_dict["beta_case"] == Case.inverse_t:
    #     params_dict["T_final"] = 50.0


class CustomArgs(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


def write_outputs(log_dir, update_dict):
    output_str = "*" * 30 + "\n"
    output_str += "Path to exp: " + log_dir + "\n"
    output_str += (
        " Custom parameters:" + json.dumps(update_dict, indent=4) + "\n"
    )
    write_to_file(SUMMARY_FILE, output_str)
    sim_file = os.path.join(log_dir, "outputs_str.txt")
    with open(sim_file, "r") as source, open(SUMMARY_FILE, "a") as destination:
        destination.write(source.read())
    output_str = "*" * 30 + "\n"
    write_to_file(SUMMARY_FILE, output_str)


def write_initial_parameters():
    if not os.path.exists(LOGGER_PATH):
        os.makedirs(LOGGER_PATH)
    params_summary = "*" * 30 + "\n"
    params_summary += (
        " Initial parameters:" + json.dumps(PARAMS, indent=4) + "\n"
    )
    params_summary += (
        " Adaptive parameters:" + json.dumps(ADAPT_PARAMS, indent=4) + "\n"
    )
    params_summary += "*" * 30 + "\n"
    write_to_file(SUMMARY_FILE, params_summary)


if __name__ == "__main__":
    combinations = list(itertools.product(*CUSTOM_PARAMS.values()))
    update_dicts = [
        {k: v for k, v in zip(CUSTOM_PARAMS.keys(), item)}
        for item in combinations
    ]

    write_initial_parameters()

    for d in update_dicts:
        # write_to_file()
        PARAMS.update(d)
        adapt_params_to_custom(PARAMS)
        PARAMS["logger_path"] = LOGGER_PATH
        args = CustomArgs(**PARAMS)
        data = parser_to_data(args)
        _, logger = run_sim(data)
        write_outputs(logger.log_dir, d)

        # args = dict_to_args(params)
        # print(["python", "main.py"] + args)
        # list_files = subprocess.run(["python", "main.py"] + args)
        # print("The exit code was: %d" % list_files.returncode)
