from pytorch_lightning.callbacks import ProgressBar
from tqdm.notebook import tqdm

def is_notebook():
    """Determine if the environment is Jupyter notebook/lab."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython
            return False
        else:
            return False  # Other type, probably standard Python interpreter
    except NameError:
        return False  # Probably standard Python interpreter


class TQDMNotebookProgressBar(ProgressBar):

    def init_train_tqdm(self):
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=2,
            disable=self.is_disabled,
            dynamic_ncols=True,
            leave=False,
            file=self.file,
            smoothing=self.smoothing,
            bar_format=self.bar_format,
            postfix=self.postfix,
            unit=self.rate_unit,
            unit_scale=self.rate_scale,
        )
        return bar

    def init_validation_tqdm(self):
        bar = tqdm(
            desc='Validating',
            position=2,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=self.file,
            smoothing=self.smoothing,
            bar_format=self.bar_format,
            postfix=self.postfix,
            unit=self.rate_unit,
            unit_scale=self.rate_scale,
        )
        return bar