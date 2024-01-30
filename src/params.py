import os
import errno
import sys
from typing import Any, Optional, IO


def dict_to_str(d, indent=0, s="-"):
    """
    Returns a string representation of a dictionary, where nested dictionaries
    are indented and lists are printed without newline characters inside them.

    Args:
        d (dict): The dictionary to convert to a string.
        indent (int): The number of spaces to indent each level of the dictionary.

    Returns:
        str: A string representation of the dictionary.
    """
    result = ""
    for key, value in d.items():
        if isinstance(value, dict):
            result += f"\n{' ' * indent}{s} {key}:"
            result += dict_to_str(value, indent + 4, s="")
        elif isinstance(value, list):
            result += f"\n{' ' * indent}{s} {key}: ["
            for i, item in enumerate(value):
                result += str(item)
                if i != len(value) - 1:
                    result += ", "
            result += "]"
        else:
            if isinstance(d[key], dict):
                result += f"\n{' ' * indent}{key}:"
                result += dict_to_str(d[key], indent + 4)
            else:
                result += f"\n{' ' * indent}{s} {key}: {value}"
    return result


class Params:
    def __init__(self, **kwargs: Any):
        """
        Initializes a Params object to store various parameters.

        This constructor initializes a default logger_path and sets additional parameters
        provided as keyword arguments.

        Args:
        kwargs: A variable number of keyword arguments.
        """
        self.logger_path = "../outputs/"
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def to_string(self) -> str:
        """
        Constructs a string representation of the parameters.

        Returns:
        str: A formatted string listing all parameters.
        """
        params_str = "-" * 50 + "  \n"
        params_str += " Params values" + "  \n"
        params_str += "-" * 50 + " "
        params_str += dict_to_str(vars(self))
        params_str += "  \n  "
        params_str += "-" * 50 + "  \n  "
        params_str += "-" * 50 + "  \n  "

        return params_str

    def print(self, f: Optional[IO] = sys.stdout) -> None:
        """
        Prints the parameters to the specified file or standard output.

        Args:
        f (Optional[IO]): The file or output stream to print the parameters to.
                          Defaults to sys.stdout.
        """
        print(self.to_string(), file=f)

    def write(self, name: str, should_print: bool = True) -> None:
        """
        Writes the parameters to a specified file and optionally prints them.

        If the directory for the file doesn't exist, it attempts to create it.

        Args:
        name (str): The file path to write the parameters to.
        should_print (bool): Flag to indicate if the parameters should also be printed.
                             Defaults to True.
        """
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with open(name, "w") as f:
            self.print(f)
        if should_print:
            self.print()
