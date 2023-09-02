#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Constants
---------------

This module contains the constants that are used in the LIONZ project.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import os

from lionz import file_utilities

# Get the root directory of the virtual environment
project_root = file_utilities.get_virtual_env_root()

# Define the paths to the trained models and the LIONZ model
NNUNET_RESULTS_FOLDER = os.path.join(project_root, 'models', 'nnunet_trained_models')
LIONZ_MODEL_FOLDER = os.path.join(NNUNET_RESULTS_FOLDER, 'nnUNet', '3d_fullres')

# Define the allowed modalities
ALLOWED_MODALITIES = ['CT', 'PT']

# Define the name of the temporary folder
TEMP_FOLDER = 'temp'

# Define color codes for console output
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RED = '\033[38;5;196m'
ANSI_RESET = '\033[0m'


# Define folder names
SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'
