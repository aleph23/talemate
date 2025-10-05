import fnmatch
import os


def list_scenes_directory(path: str = ".") -> list:

    """
    List all the scene files in the given directory.
    :param directory: Directory to list scene files from.
    :return: List of scene files in the given directory.
    """ 
    current_dir = os.getcwd()

    scenes = _list_files_and_directories(os.path.join(current_dir, "scenes"), path)

    return scenes


def _list_files_and_directories(root: str, path: str) -> list:
    # Define the file patterns to match
    """List all the files and directories in the given root directory.
    
    This function traverses the directory tree starting from the specified  root
    directory. It collects files that match predefined patterns while  excluding
    JSON files located in 'nodes' directories. The function uses  `os.walk` to
    navigate through the directory structure and applies  `fnmatch` to filter the
    relevant files based on the specified patterns.
    
    Args:
        root (str): Root directory to list files and directories from.
        path (str): Relative path to list files and directories from.
    """
    patterns = ["characters/*.png", "characters/*.webp", "*/*.json"]

    items = []

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Check each file if it matches any of the patterns
        for filename in filenames:
            # Skip JSON files inside 'nodes' directories
            if filename.endswith(".json") and "nodes" in dirpath.split(os.sep):
                continue

            # Get the relative file path
            rel_path = os.path.relpath(dirpath, root)
            for pattern in patterns:
                if fnmatch.fnmatch(os.path.join(rel_path, filename), pattern):
                    items.append(os.path.join(dirpath, filename))
                    break

    return items
