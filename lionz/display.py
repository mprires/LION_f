#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Display Messages
-----------------------

Author: Lalith Kumar Shiyam Sundar, Sebastian Gutschmayer, Manuel pires
Institution: Medical University of Vienna
Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
Date: 09.02.2023
Version: 0.1.0

This module provides predefined display messages for the LIONZ application.

Usage:
    The functions in this module can be imported and used in other modules within
    the LIONZ to show predefined display messages. Typical usage might include::

        from lionz.display import logo, citation

        # Display the LIONZ logo:
        logo()

"""

import logging
import emoji
import pyfiglet

from lionz import constants
from lionz import resources


def get_usage_message():
    """
    Get the usage message for LIONZ.

    :return: str: A message detailing the usage instructions for the LIONZ application.
    """
    usage_message = """
    Usage:
      lionz -d <MAIN_DIRECTORY> -m <MODEL_NAME>
    Example:  
      lionz -d /Documents/Data_to_lionz/ -m clin_ct_lesions

    Description:
      LIONZ (Lesion segmentatION) - A state-of-the-art AI solution that
      emphasizes precise lesion segmentation in diverse imaging datasets.
    """
    return usage_message


def logo():
    """
    Display the LIONZ logo.

    This function presents the LIONZ logo using the pyfiglet library and ANSI color codes.
    """
    print(' ')
    logo_color_code = constants.ANSI_VIOLET
    slogan_color_code = constants.ANSI_VIOLET
    result = logo_color_code + pyfiglet.figlet_format(" LIONZ 0.1.0", font="smslant").rstrip() + "\033[0m"
    text = slogan_color_code + " A part of the ENHANCE community. Join us at www.enhance.pet to build the future of " \
                               "medical imaging together." + "\033[0m"
    print(result)
    print(text)
    print(' ')


def citation():
    """
    Display the manuscript citation for LIONZ.

    This function offers the manuscript citation for the LIONZ project.
    """
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":scroll:")} CITATION:{constants.ANSI_RESET}')
    print(" ")
    print(
        " Shiyam Sundar LK, Gutschmayer S, pires M, et al. LIONZ: Advancing lesion segmentation "
        "in medical imaging datasets. J Nucl Med. Forthcoming 2023.")
    print(" Copyright 2023, Quantitative Imaging and Medical Physics Team, Medical University of Vienna")
