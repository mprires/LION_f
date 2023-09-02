#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions that validate the user inputs for the LIONz project.

It checks parameters like the existence of the parent folder and the validity of the model name.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>, 

.. versionadded:: 0.1.0
"""

import os
import logging
import emoji
from lionz.resources import AVAILABLE_MODELS
from lionz import constants


def validate_inputs(parent_folder: str, model_name: str) -> bool:
    """
    Validates the user inputs for the main function.
    
    :param parent_folder: The parent folder containing subject folders.
    :type parent_folder: str
    
    :param model_name: The name of the model to use for segmentation.
    :type model_name: str
    
    :return: True if the inputs are valid, False otherwise.
    :rtype: bool
    """
    return validate_parent_folder(parent_folder) and validate_model_name(model_name)


def validate_parent_folder(parent_folder: str) -> bool:
    """Validates if the parent folder exists."""
    if os.path.isdir(parent_folder):
        return True
    else:
        message = f"The parent folder {parent_folder} does not exist."
        logging.error(message)
        print_error(message)
        return False


def validate_model_name(model_name: str) -> bool:
    """Validates if the model name is available."""
    if model_name in AVAILABLE_MODELS:
        return True
    else:
        message = f"The model name {model_name} is invalid."
        logging.error(message)
        print_error(message)
        return False


def print_error(message: str):
    """Prints an error message with standard formatting."""
    print(f"{emoji.emojize(':cross_mark:')} {constants.ANSI_RED} {message} {constants.ANSI_RESET}")


# Remember to replace <email_of_sebastian> and <email_of_manuel> with the actual email addresses.
