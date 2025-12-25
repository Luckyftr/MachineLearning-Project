from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_path(*args):
    
    return PROJECT_ROOT.joinpath(*args)
