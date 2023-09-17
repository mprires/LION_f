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
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool

import nibabel as nib

from lionz.constants import SEGMENTATIONS_FOLDER, STATS_FOLDER, WORKFLOW_FOLDER
from lionz.resources import TRACER_WORKFLOWS


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


def lion_folder_structure(parent_directory: str, model_name: str, modalities: list) -> tuple:
    """
    Creates the lion folder structure.

    :param parent_directory: The path to the parent directory.
    :type parent_directory: str

    :param model_name: The name of the model.
    :type model_name: str

    :param modalities: The list of modalities.
    :type modalities: list

    :return: A tuple containing the paths to the lion directory, input directories, output directory, and stats directory.
    :rtype: tuple
    """
    lion_dir = os.path.join(parent_directory,
                            'lionz-' + model_name + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    create_directory(lion_dir)
    input_dirs = []
    for modality in modalities:
        input_dirs.append(os.path.join(lion_dir, modality))
        create_directory(input_dirs[-1])

    output_dir = os.path.join(lion_dir, SEGMENTATIONS_FOLDER)
    stats_dir = os.path.join(lion_dir, STATS_FOLDER)
    workflow_dir = os.path.join(lion_dir, WORKFLOW_FOLDER)
    create_directory(output_dir)
    create_directory(stats_dir)
    create_directory(workflow_dir)
    return lion_dir, input_dirs, output_dir, stats_dir, workflow_dir


def organise_files_by_modality(lion_compliant_subjects: list, modalities: list, lion_dir: str) -> None:
    """
    Organises the files by modality.

    :param lion_compliant_subjects: The list of lion-compliant subjects paths.
    :type lion_compliant_subjects: list

    :param modalities: The list of modalities.
    :type modalities: list

    :param lion_dir: The path to the lion directory.
    :type lion_dir: str
    """
    for modality in modalities:
        files_to_copy = select_files_by_modality(lion_compliant_subjects, modality)
        copy_files_to_destination(files_to_copy, os.path.join(lion_dir, modality))


def copy_files_to_destination(files: list, destination: str) -> None:
    """
    Copies the files inside the list to the destination directory in a parallel fashion.

    :param files: The list of files to be copied.
    :type files: list

    :param destination: The path to the destination directory.
    :type destination: str
    """
    with Pool(processes=len(files)) as pool:
        pool.starmap(copy_file, [(file, destination) for file in files])


def select_files_by_modality(lion_compliant_subjects: list, modality_tag: str) -> list:
    """
    Selects the files with the selected modality tag from the lion-compliant folders.

    :param lion_compliant_subjects: The list of lion-compliant subjects paths.
    :type lion_compliant_subjects: list

    :param modality_tag: The modality tag to be selected.
    :type modality_tag: str

    :return: The list of selected files.
    :rtype: list
    """
    selected_files = []
    for subject in lion_compliant_subjects:
        files = os.listdir(subject)
        for file in files:
            if file.startswith(modality_tag) and (file.endswith('.nii') or file.endswith('.nii.gz')):
                selected_files.append(os.path.join(subject, file))
    return selected_files


def copy_file(file: str, destination: str) -> None:
    """
    Copies a file to the specified destination.

    :param file: The path to the file to be copied.
    :type file: str

    :param destination: The path to the destination directory.
    :type destination: str
    """
    shutil.copy(file, destination)


def create_model_based_workflows(lion_dir: str, model_name: str):
    """
    Organises the files from the modality directories (CT, PT) to their respective workflow folders
    and converts them into a nnU-Net compatible format based on the provided model_name.

    :param lion_dir: Directory path where the subject's modalities are stored.
    :type lion_dir: str

    :param model_name: The model name, e.g., 'fdg' or 'psma'.
    :type model_name: str
    """

    # Ensure the model_name is valid
    if model_name not in TRACER_WORKFLOWS:
        raise ValueError(f"Unknown model_name: {model_name}")

    for workflow_name, workflow_data in TRACER_WORKFLOWS[model_name]['workflows'].items():
        workflow_dir = os.path.join(lion_dir, WORKFLOW_FOLDER, f'{model_name}_{workflow_name}')
        create_directory(workflow_dir)

        for modality, suffix in workflow_data['channels'].items():
            modality_dir = os.path.join(lion_dir, modality)

            for file in os.listdir(modality_dir):
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    base_name = os.path.basename(workflow_dir)

                    new_filename = f"{base_name}_{suffix}"

                    img = nib.load(os.path.join(modality_dir, file))
                    nib.save(img, os.path.join(workflow_dir, new_filename))
