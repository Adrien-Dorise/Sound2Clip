"""
This package references all utils for path verifications
Author: Adrien Dorise (adrien.dorise@hotmail.com), Edouard Villain (evillain@lrtechnologies.fr) - LR Technologies
Created: June 2024
Last updated: Adrien Dorise - June 2024
"""
import os

def create_file_path(path):
    """Check if a file path exists. 
    If not, the path to the folder storing the file is created

    Args:
        path (str): Path to a file (ex: my/folder/my_file)
    """

    file_name = path.split("/")[-1]
    folder_path = path[0:-len(file_name)]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)