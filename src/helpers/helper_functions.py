from pathlib import Path
import shutil

def safe_mkdir(path):
    """Ensure, that the folder exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_mkdir_clean(path):
    """Ensure that the folder exists and delete its content"""
    safe_mkdir(path)
    shutil.rmtree(path)
    safe_mkdir(path) 