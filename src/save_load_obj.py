import dill as pickle
from typing import Any


def save_obj(obj: Any, path: str) -> None:
    """
    Save a Python object to a file using the dill module.

    The function opens the specified file in write-binary mode and uses dill's dump method
    to serialize and save the object. After saving, it closes the file.

    Args:
    obj (Any): The Python object to be saved.
    path (str): The file path where the object will be saved.

    Returns:
    None
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)
        print(f"- Obj saved as {path}")


def load_obj(path: str) -> Any:
    """
    Load a Python object from a file using the dill module.

    The function opens the specified file in read-binary mode and uses dill's load method
    to deserialize and load the object. After loading, it closes the file and returns the object.

    Args:
    path (str): The file path from which the object will be loaded.

    Returns:
    Any: The deserialized Python object.
    """
    with open(path, "rb") as file_to_read:
        obj = pickle.load(file_to_read)
        print(f"- Obj loaded from {path}")
        return obj
