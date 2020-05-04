import git
import sys


def init():
    root_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    sys.path.insert(1, root_dir)
    global CONFIG_FILE
    CONFIG_FILE = f'{root_dir}/modcovid/config.yml'