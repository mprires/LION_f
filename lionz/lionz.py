#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIONZ: Lesion Segmentation Tool
-------------------------------

This module, `lionz.py`, serves as the main entry point for the LIONZ toolkit.
It provides capabilities for tumor and lesion segmentation in PET/CT datasets.

Notes
-----
.. note:: 
   For a full understanding of the capabilities and functionalities of this module, 
   refer to the individual function and class docstrings.

Attributes
----------
__author__ : str
    Module author(s).
    
__email__ : str
    Contact email for module inquiries.

__version__ : str
    Current version of the module.

Examples
--------
To use this module, you can either import it into another script or run it directly:

.. code-block:: python

    import lionz
    # Use functions or classes

or:

.. code-block:: bash

    $ python lionz.py

See Also
--------
constants : Module containing constant values used throughout the toolkit.
display : Module responsible for displaying information and graphics.
image_processing : Module with functions and classes for image processing tasks.
input_validation : Module that provides functionalities for validating user inputs.
resources : Contains resource files and data necessary for the toolkit.
download : Handles downloading of data, models, or other necessary resources.

"""

__author__ = "Lalith kumar shiyam sundar, Sebastian Gutschmayer, Manuel pires"
__email__ = "lalith.shiyamsundar@meduniwien.ac.at, sebastian.gutschmayer@meduniwien.ac.at, manuel.pires@meduniwien.ac.at"
__version__ = "0.1"


# Imports for the module


import argparse
import glob
import logging
import os
import time
import emoji
from datetime import datetime
import colorama

from lionz import display
from lionz import constants
from lionz.resources import AVAILABLE_MODELS

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                    filename=datetime.now().strftime('lionz-v.0.1.0.%H-%M-%d-%m-%Y.log'),
                    filemode='w')

# Main function for the module
def main():
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
    description=display.get_usage_message(),
    formatter_class=argparse.RawTextHelpFormatter,  # To retain the custom formatting
    add_help=False  # We'll add our own help option later
    )

    # Main directory containing subject folders
    parser.add_argument(
        "-d", "--main_directory", 
        type=str, 
        required=True,
        metavar="<MAIN_DIRECTORY>",
        help="Specify the main directory containing subject folders."
    )
    
    # Name of the model to use for segmentation
    model_help_text = "Choose the model for segmentation from the following:\n" + "\n".join(AVAILABLE_MODELS)
    parser.add_argument(
        "-m", "--model_name", 
        type=str, 
        choices=AVAILABLE_MODELS, 
        required=True,
        metavar="<MODEL_NAME>",
        help=model_help_text
    )
    
    # Custom help option
    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )


    args = parser.parse_args()

    parent_folder = os.path.abspath(args.main_directory)
    model_name = args.model_name

    display.logo()
    display.citation()