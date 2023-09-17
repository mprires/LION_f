import site
import os
from lionz import file_utilities


def add_custom_trainers_to_local_nnunetv2():
    # Locate the site-packages directory
    site_packages = site.getsitepackages()[0]

    source_file_path = os.path.join(site_packages, 'lionz', 'nnUNet_custom_trainer', 'LION_custom_trainers.py')
    target_file_path = os.path.join(site_packages, 'nnunetv2', 'training', 'nnUNetTrainer', 'variants',
                                    'LION_custom_trainers.py')

    # Check if the file exists
    if not os.path.exists(source_file_path):
        return f'Could not find custom trainer file: {source_file_path}'

    # Check if the file exists
    if os.path.exists(target_file_path):
        return f'Custom trainer already installed: {target_file_path}.'

    file_utilities.copy_file(source_file_path, target_file_path)

    return f"Custom trainer added to {target_file_path}"
