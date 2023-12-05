import importlib


class GlobalDataType:
    data_type = None

    @staticmethod
    def set_global_data_type(config_file):
        config_module = importlib.import_module(args.config_file.replace("/", "."))
        params = getattr(config_module, "CONFIG")
