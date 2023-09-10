import os
import errno
import sys


def dict_to_str(d, indent=0, s="-"):
    """
    Returns a string representation of a dictionary, where nested dictionaries are indented and
    lists are printed without newline characters inside them.

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


class Data:
    def __init__(self, **kwargs):
        """General data object to store parameters."""
        # for key, value in INIT_DICT.items():
        #     self.__setattr__(key, value)
        self.logger_path = "outputs/tensorboard_logs"
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def to_string(self):
        """Print parameters.

        Returns:
            A string of the parameters to be printed.
        """
        s = ""
        s += "-" * 50 + "  \n"
        s += "Data values" + "  \n"
        s += "-" * 50 + " "
        s += dict_to_str(vars(self))
        s += "  \n  "
        s += "-" * 50 + "  \n  "
        s += "-" * 50 + "  \n  "

        return s

    def print(self, f=sys.stdout):
        """Print the data.

        Args:
            f: Where to print data. Defaults to sys.stdout.
        """
        print(self.to_string(), file=f)

    def write(self, name, print=True):
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        f = open(name, "w")
        self.print(f)
        if print:
            self.print(None)
        f.close()
