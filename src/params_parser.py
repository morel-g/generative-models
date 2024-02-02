import argparse
from src.case import Case


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Custom formatter for argparse help text, extending the default HelpFormatter.

    This formatter adjusts the help text's formatting, specifically the position and
    width of help messages, and the format of command-line option descriptions.

    Args:
    prog (str): The name of the program (typically sys.argv[0]).
    """

    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=40, width=80)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        """
        Formats the action invocation part of the help message.

        If the action has no option strings (like positional arguments) or if its nargs
        is 0, it uses the default formatting. Otherwise, it customizes the format of
        option strings.

        Args:
        action (argparse.Action): The action to format.

        Returns:
        str: The formatted action invocation string.
        """
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string


def init_parser(description: str) -> argparse.ArgumentParser:
    """
    Initializes and returns an argparse.ArgumentParser with a custom help formatter.

    This function creates a parser with a specified description and a custom help
    formatter, which adjusts the help text's formatting.

    Args:
    description (str): A description of the program for which the parser is being created.

    Returns:
    argparse.ArgumentParser: The initialized argument parser.
    """
    return argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter,
        description=description,
        epilog="",
    )


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
    general.add_argument("--no_adapt_dt", dest="adapt_dt", action="store_false")
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
        "--save_samples",
        action="store_true",
        default=False,
        help="Save samples from the model.",
    )
    general.add_argument(
        "--nb_batch_saved",
        type=int,
        default=1,
        metavar="",
        help="Number of batch saved. Only used when save_samples is True.",
    )
    general.add_argument(
        "--save_noise",
        action="store_true",
        default=False,
        help="Save original noise from the prior.  Only used when save_samples is True.",
    )
    general.add_argument(
        "--no_ema",
        action="store_true",
        default=False,
        help="Do not use ema model for viz.",
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
        default=[5, 5],
        metavar="",
        help="Number of rows on columns for images display.",
    )
    return parser.parse_args()
