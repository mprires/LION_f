#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions that validate the user inputs for the LIONz project.

It checks parameters like the existence of the parent folder and the validity of the model name.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>, 

.. versionadded:: 0.1.0
"""

import logging
import os
from typing import List

import emoji

from lionz import constants
from lionz.resources import AVAILABLE_MODELS


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


def select_lion_compliant_subjects(subject_paths: List[str], modality_tags: List[str]) -> List[str]:
    """
    Selects the subjects that have the files that have names that are compliant with the moosez.

    :param subject_paths: The path to the list of subjects that are present in the parent directory.
    :type subject_paths: List[str]
    :param modality_tags: The list of appropriate modality prefixes that should be attached to the files for them to be moose compliant.
    :type modality_tags: List[str]
    :return: The list of subject paths that are moose compliant.
    :rtype: List[str]
    """
    # go through each subject in the parent directory
    lion_compliant_subjects = []
    for subject_path in subject_paths:
        # go through each subject and see if the files have the appropriate modality prefixes

        files = [file for file in os.listdir(subject_path) if file.endswith('.nii') or file.endswith('.nii.gz')]
        prefixes = [file.startswith(tag) for tag in modality_tags for file in files]
        if sum(prefixes) == len(modality_tags):
            lion_compliant_subjects.append(subject_path)
    print(f"{constants.ANSI_ORANGE} Number of lion compliant subjects: {len(lion_compliant_subjects)} out of "
          f"{len(subject_paths)} {constants.ANSI_RESET}")
    logging.info(f" Number of lion compliant subjects: {len(lion_compliant_subjects)} out of "
                 f"{len(subject_paths)}")

    return lion_compliant_subjects
