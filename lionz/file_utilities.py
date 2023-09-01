#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ File Utilities
---------------------

This module provides utility functions specifically designed to handle file operations for the LIONZ application.

LIONZ, standing for Lesion segmentatION, offers an advanced solution for lesion segmentation tasks within medical imaging datasets. The file utilities module ensures efficient, reliable, and organized manipulation of files and directories—be it reading, writing, or organizing data, configuration files, model artifacts, and more. Such functions simplify the I/O operations and maintain the consistency and integrity of the application's data structure.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import os 
import sys

def get_virtual_env_root() -> str:
    """
    Returns the root directory of the virtual environment.
    
    :return: The root directory of the virtual environment.
    :rtype: str
    """
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


def get_files(directory: str, wildcard: str) -> list:
    """
    Returns the sorted list of files in the directory with the specified wildcard.
    
    :param directory: The directory path.
    :type directory: str
    
    :param wildcard: The wildcard to be used.
    :type wildcard: str
    
    :return: The sorted list of files.
    :rtype: list
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(wildcard):
            files.append(os.path.join(directory, file))
    return sorted(files)


def create_directory(directory_path: str) -> str:
    """
    Creates a directory at the specified path and returns its path.
    
    :param directory_path: The path to the directory.
    :type directory_path: str
    
    :return: The path of the created directory.
    :rtype: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    return directory_path

