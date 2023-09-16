#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Resources
---------------

This module contains functions that are responsible for downloading and managing resources for the LIONZ application.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import io
import logging
import os
import zipfile

import requests
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, FileSizeColumn, TransferSpeedColumn, TimeRemainingColumn

from lionz import constants
from lionz import resources


def model(tracer_name, model_path):
    """
    Downloads the models for the specified tracer for the current system.

    :param tracer_name: The name of the tracer (e.g., "fdg", "psma").
    :type tracer_name: str
    :param model_path: The path to store the models.
    :type model_path: str
    """
    models_list = resources.MODELS[tracer_name]
    total_downloads = len(models_list)

    for idx, model_info in enumerate(models_list, start=1):
        url = model_info["url"]
        filename = os.path.join(model_path, model_info["filename"])
        directory = os.path.join(model_path, model_info["directory"])

        if not os.path.exists(directory):
            logging.info(f"Downloading {directory}")

            # Get the total size for the download
            response = requests.get(url, stream=True)
            download_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024 * 10

            # Get the total size for the extraction
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                extraction_size = sum((file.file_size for file in zip_ref.infolist()))

            total_size = download_size + extraction_size

            console = Console()
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                FileSizeColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=True
            )

            with progress:
                task_desc = f"[white] Downloading {tracer_name} model [{idx}/{total_downloads}]..."
                task = progress.add_task(task_desc, total=total_size)
                for chunk in response.iter_content(chunk_size=chunk_size):
                    open(filename, "ab").write(chunk)
                    progress.update(task, advance=chunk_size)

                # Unzip the model
                parent_directory = os.path.dirname(directory)
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    for file in zip_ref.infolist():
                        zip_ref.extract(file, parent_directory)
                        extracted_size = file.file_size
                        progress.update(task, advance=extracted_size)

            logging.info(f"{os.path.basename(directory)} extracted.")

            # Delete the zip file
            os.remove(filename)
            print(f"{constants.ANSI_GREEN} {os.path.basename(directory)} - download complete. {constants.ANSI_RESET}")
            logging.info(f"{os.path.basename(directory)} - download complete.")
        else:
            print(f"{constants.ANSI_GREEN} A local instance of {os.path.basename(directory)} has been detected. {constants.ANSI_RESET}")
            logging.info(f"A local instance of {os.path.basename(directory)} has been detected.")
