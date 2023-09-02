#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Resources
---------------

This module contains utility functions and resources that are crucial for the operations of the LIONZ application.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets. The resources module is designed to manage and provide auxiliary resources, such as configuration files, model weights, and other important artifacts necessary for the proper functioning of the application.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import torch
from lionz import constants 


# List of available models in the LIONZ application
AVAILABLE_MODELS = ["fdg_lionz",
                    "psma_lionz"]

# Dictionary of expected modalities for each model in the LIONZ application
EXPECTED_MODALITIES = {"fdg_lionz": ["PT", "CT"],
                          "psma_lionz": ["PT"]}


def check_cuda() -> str:
    """
    This function checks if CUDA is available on the device and prints the device name and number of CUDA devices
    available on the device.

    Returns:
        str: The device to run predictions on, either "cpu" or "cuda".
    """
    if not torch.cuda.is_available():
        print(
            f"{constants.ANSI_ORANGE}CUDA not available on this device. Predictions will be run on CPU.{constants.ANSI_RESET}")
        return "cpu"
    else:
        device_count = torch.cuda.device_count()
        print(
            f"{constants.ANSI_GREEN} CUDA is available on this device with {device_count} GPU(s). Predictions will be run on GPU.{constants.ANSI_RESET}")
        return "cuda"