import argparse
from ..case import Case


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=80)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string


def init_parser(description: str) -> argparse.ArgumentParser:
    """Initializes a parser with the given description and custom formatter."""
    return argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter,
        description=description,
        epilog="",
    )


def parse_main() -> argparse.Namespace:
    """Parses arguments for the main parser.

    Returns:
        Parsed arguments.
    """
    parser = init_parser(
        description="Train a generative model with a specific config file."
    )

    # General parser
    general = parser.add_argument_group("General options")
    general.add_argument(
        "--config_file",
        type=str,
        default=None,
        metavar="",
        help="The config file used during for the training.",
    )
    general.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=None,
        metavar="",
        help="GPU id used.",
    )

    return parser.parse_args()


def parse_viz() -> argparse.Namespace:
    """Parses arguments for the visualization parser.

    Returns:
        Parsed arguments.
    """
    parser = init_parser(description="Training parser.")

    # General parser
    general = parser.add_argument_group("General options")
    general.add_argument(
        "-c",
        "--ckpt_path",
        type=str,
        default="",
        help="Chekpoint path.",
    )
    general.add_argument("--adapt_dt", action="store_true")
    general.add_argument(
        "--no_adapt_dt", dest="adapt_dt", action="store_false"
    )
    general.set_defaults(adapt_dt=None)
    general.add_argument(
        "-ts",
        "--nb_time_steps_eval",
        type=int,
        default=None,
        metavar="",
        help="Number of time steps used for evaluation. Same as during training by default.",
    )
    general.add_argument(
        "-bs",
        "--batch_size_eval",
        type=int,
        default=None,
        metavar="",
        help="Batch size used for evaluation.",
    )
    general.add_argument(
        "-tv",
        "--nb_time_validation",
        type=int,
        default=None,
        metavar="",
        help="Number of time where the validation loss is computed with different values for the time variables. Used only when loss is set to True",
    )
    general.add_argument(
        "-scheme",
        "--scheme",
        type=str,
        default=None,
        metavar="",
        help="Backward scheme used for evaluation.",
    )
    general.add_argument(
        "-l",
        "--loss",
        action="store_true",
        default=False,
        help="Compute loss.",
    )
    general.add_argument(
        "-f",
        "--fid",
        action="store_true",
        default=False,
        help="Compute fid.",
    )
    general.add_argument(
        "-fc",
        "--fid_choice",
        type=str,
        choices=[Case.fid_v1, Case.fid_v3, Case.fid_metrics_v3],
        default=Case.fid_metrics_v3,
        help="Select a version from fid evaluation.",
    )
    general.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        default=0,
        metavar="",
        help="GPU id used, -1 for CPU.",
    )
    general.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="Additional name for the plots.",
    )
    general.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        metavar="",
        help="Output dir to save the figures.",
    )
    general.add_argument(
        "--nb_imgs",
        nargs="+",
        type=int,
        default=[3, 3],
        metavar="",
        help="Number of rows on columns for images display.",
    )
    return parser.parse_args()
