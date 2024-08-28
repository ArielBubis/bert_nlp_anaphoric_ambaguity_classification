# hook-transformers.py
from PyInstaller.utils.hooks import collect_submodules

# Collect only the necessary submodules
hiddenimports = [
    'transformers.models.distilbert.modeling_distilbert',
    'transformers.models.distilbert.tokenization_distilbert'
    ]