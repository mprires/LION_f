#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ image processing
---------------

This module contains functions that are responsible for image processing in LION.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""
import os
import csv
import SimpleITK as sitk
import dask.array as da
import nibabel
import numpy as np
import cv2
import imageio
import logging

from scipy.ndimage import rotate
from skimage import exposure
from dask.distributed import Client

from lionz.constants import CHUNK_THRESHOLD, MATRIX_THRESHOLD, Z_AXIS_THRESHOLD, INTERPOLATION,  MIP_VOXEL_SPACING, \
    FRAME_DURATION
from typing import Tuple

class NiftiPreprocessor:
    """
    A class for processing NIfTI images using nibabel and SimpleITK.

    Attributes:
    -----------
    image: nibabel.Nifti1Image
        The NIfTI image to be processed.
    original_header: nibabel Header
        The original header information of the NIfTI image.
    is_large: bool
        Flag indicating if the image is classified as large.
    sitk_image: SimpleITK.Image
        The image converted into a SimpleITK object.
    """

    def __init__(self, image: nibabel.Nifti1Image):
        """
        Constructs all the necessary attributes for the NiftiPreprocessor object.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to be processed.
        """
        self.image = image
        self.original_header = image.header.copy()
        self.is_large = self._is_large_image(image.shape)
        self.sitk_image = self._convert_to_sitk(self.image)

    @staticmethod
    def _is_large_image(image_shape) -> bool:
        """
        Check if the image classifies as large based on pre-defined thresholds.

        Parameters:
        -----------
        image_shape: tuple
            The shape of the NIfTI image.

        Returns:
        --------
        bool
            True if the image is large, False otherwise.
        """
        return np.prod(image_shape) > MATRIX_THRESHOLD and image_shape[2] > Z_AXIS_THRESHOLD

    @staticmethod
    def _is_orthonormal(image: nibabel.Nifti1Image) -> bool:
        """
        Check if the qform or sform of a NIfTI image is orthonormal.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to check.

        Returns:
        --------
        bool
            True if qform or sform is orthonormal, False otherwise.
        """
        # Check qform
        qform_code = image.header["qform_code"]
        if qform_code != 0:
            qform = image.get_qform()
            q_rotation = qform[:3, :3]
            q_orthonormal = np.allclose(np.dot(q_rotation, q_rotation.T), np.eye(3))
            # if not q_orthonormal:
            # return False

        # Check sform
        sform = image.get_sform()
        s_rotation = sform[:3, :3]
        s_orthonormal = np.allclose(np.dot(s_rotation, s_rotation.T), np.eye(3))
        if not s_orthonormal:
            return False

        return True

    @staticmethod
    def _make_orthonormal(image: nibabel.Nifti1Image) -> nibabel.Nifti1Image:
        """
        Make a NIFTI image orthonormal while keeping the diagonal of the rotation matrix positive.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to make orthonormal.

        Returns:
        --------
        nibabel.Nifti1Image
            The orthonormal NIFTI image.
        """
        new_affine = image.affine
        new_header = image.header

        rotation_scaling = new_affine[:3, :3]

        q, r = np.linalg.qr(rotation_scaling)
        diagonal_sign = np.sign(np.diag(r))
        q = q @ np.diag(diagonal_sign)
        orthonormal = q

        new_affine[:3, :3] = orthonormal
        new_header['pixdim'][1:4] = np.diag(orthonormal)
        new_header['srow_x'] = new_affine[0, :]
        new_header['srow_y'] = new_affine[1, :]
        new_header['srow_z'] = new_affine[2, :]

        new_image = nibabel.Nifti1Image(image.get_fdata(), affine=new_affine, header=new_header)

        return new_image

    @staticmethod
    def _convert_to_sitk(image: nibabel.Nifti1Image) -> sitk.Image:
        """
        Convert a NIfTI image to a SimpleITK image, retaining the original header information.

        Parameters:
        -----------
        image: nibabel.Nifti1Image
            The NIfTI image to convert.

        Returns:
        --------
        sitk.Image
            The SimpleITK image.
        """
        image_data = image.get_fdata()
        image_affine = image.affine
        original_spacing = image.header.get_zooms()

        image_data_swapped_axes = image_data.swapaxes(0, 2)
        sitk_image = sitk.GetImageFromArray(image_data_swapped_axes)

        translation_vector = image_affine[:3, 3]
        rotation_matrix = image_affine[:3, :3]
        axis_flip_matrix = np.diag([-1, -1, 1])

        sitk_image.SetSpacing([spacing.item() for spacing in original_spacing])
        sitk_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
        sitk_image.SetDirection((np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).flatten())

        return sitk_image


class ImageResampler:
    @staticmethod
    def chunk_along_axis(axis: int) -> int:
        """
        Determines the maximum number of evenly-sized chunks that the axis can be split into.
        Each chunk is at least of size CHUNK_THRESHOLD.

        :param axis: Length of the axis.
        :type axis: int
        :return: The maximum number of evenly-sized chunks.
        :rtype: int
        :raises ValueError: If axis is negative or if CHUNK_THRESHOLD is less than or equal to 0.
        """
        # Check for negative input values
        if axis < 0:
            raise ValueError('Axis must be non-negative')

        if CHUNK_THRESHOLD <= 0:
            raise ValueError('CHUNK_THRESHOLD must be greater than 0')

        # If the axis is smaller than the threshold, it cannot be split into smaller chunks
        if axis < CHUNK_THRESHOLD:
            return 1

        # Determine the maximum number of chunks that the axis can be split into
        split = axis // CHUNK_THRESHOLD

        # Reduce the number of chunks until axis is evenly divisible by split
        while axis % split != 0:
            split -= 1

        return split

    @staticmethod
    def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: int,
                                 output_spacing: tuple, output_size: tuple) -> da.array:
        """
        Resamples a dask array chunk.

        :param image_chunk: The chunk (part of an image) to be resampled.
        :type image_chunk: da.array
        :param input_spacing: The original spacing of the chunk (part of an image).
        :type input_spacing: tuple
        :param interpolation_method: SimpleITK interpolation type.
        :type interpolation_method: int
        :param output_spacing: Spacing of the newly resampled chunk.
        :type output_spacing: tuple
        :param output_size: Size of the newly resampled chunk.
        :type output_size: tuple
        :return: The resampled chunk (part of an image).
        :rtype: da.array
        """
        sitk_image_chunk = sitk.GetImageFromArray(image_chunk)
        sitk_image_chunk.SetSpacing(input_spacing)
        input_size = sitk_image_chunk.GetSize()

        if all(x == 0 for x in input_size):
            return image_chunk

        resampled_sitk_image = sitk.Resample(sitk_image_chunk, output_size, sitk.Transform(),
                                             interpolation_method,
                                             sitk_image_chunk.GetOrigin(), output_spacing,
                                             sitk_image_chunk.GetDirection(), 0.0, sitk_image_chunk.GetPixelIDValue())

        resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)
        return resampled_array

    @staticmethod
    def resample_image_SimpleITK_DASK(sitk_image: sitk.Image, interpolation: str,
                                      output_spacing: tuple = (1.5, 1.5, 1.5),
                                      output_size: tuple = None) -> sitk.Image:
        """
        Resamples a sitk_image using Dask and SimpleITK.

        :param sitk_image: The SimpleITK image to be resampled.
        :type sitk_image: sitk.Image
        :param interpolation: nearest|linear|bspline.
        :type interpolation: str
        :param output_spacing: The desired output spacing of the resampled sitk_image.
        :type output_spacing: tuple
        :param output_size: The new size to use.
        :type output_size: tuple
        :return: The resampled sitk_image as SimpleITK.Image.
        :rtype: sitk.Image
        :raises ValueError: If the interpolation method is not supported.
        """
        if interpolation == 'nearest':
            interpolation_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation == 'bspline':
            interpolation_method = sitk.sitkBSpline
        else:
            raise ValueError('The interpolation method is not supported.')

        input_spacing = sitk_image.GetSpacing()
        input_size = sitk_image.GetSize()
        input_chunks = (input_size[0] / ImageResampler.chunk_along_axis(input_size[0]),
                        input_size[1] / ImageResampler.chunk_along_axis(input_size[1]),
                        input_size[2] / ImageResampler.chunk_along_axis(input_size[2]))
        input_chunks_reversed = list(reversed(input_chunks))

        image_dask = da.from_array(sitk.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)

        if output_size is not None:
            output_spacing = [input_spacing[i] * (input_size[i] / output_size[i]) for i in range(len(input_size))]

        output_chunks = [round(input_chunks[i] * (input_spacing[i] / output_spacing[i])) for i in
                         range(len(input_chunks))]
        output_chunks_reversed = list(reversed(output_chunks))

        result = da.map_blocks(ImageResampler.resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method,
                               output_spacing, output_chunks, chunks=output_chunks_reversed)

        resampled_image = sitk.GetImageFromArray(result)
        resampled_image.SetSpacing(output_spacing)
        resampled_image.SetOrigin(sitk_image.GetOrigin())
        resampled_image.SetDirection(sitk_image.GetDirection())

        return resampled_image

    @staticmethod
    def resample_image_SimpleITK(sitk_image: sitk.Image, interpolation: str,
                                 output_spacing: tuple = (1.5, 1.5, 1.5),
                                 output_size: tuple = None) -> sitk.Image:
        """
        Resamples an image to a new spacing using SimpleITK.

        :param sitk_image: The input image.
        :type sitk_image: SimpleITK.Image
        :param interpolation: The interpolation method to use. Supported methods are 'nearest', 'linear', and 'bspline'.
        :type interpolation: str
        :param output_spacing: The new spacing to use. Default is (1.5, 1.5, 1.5).
        :type output_spacing: tuple
        :param output_size: The new size to use. Default is None.
        :type output_size: tuple
        :return: The resampled image as SimpleITK.Image.
        :rtype: SimpleITK.Image
        :raises ValueError: If the interpolation method is not supported.
        """
        if interpolation == 'nearest':
            interpolation_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation == 'bspline':
            interpolation_method = sitk.sitkBSpline
        else:
            raise ValueError('The interpolation method is not supported.')

        desired_spacing = np.array(output_spacing).astype(np.float64)
        if output_size is None:
            input_size = sitk_image.GetSize()
            input_spacing = sitk_image.GetSpacing()
            output_size = [round(input_size[i] * (input_spacing[i] / output_spacing[i])) for i in
                           range(len(input_size))]

        # Interpolation:
        resampled_sitk_image = sitk.Resample(sitk_image, output_size, sitk.Transform(), interpolation_method,
                                             sitk_image.GetOrigin(), desired_spacing,
                                             sitk_image.GetDirection(), 0.0, sitk_image.GetPixelIDValue())

        return resampled_sitk_image

    @staticmethod
    def resample_image(moose_img_object, interpolation: str, desired_spacing: tuple,
                       desired_size: tuple = None) -> nibabel.Nifti1Image:
        """
        Resamples an image to a new spacing.

        :param moose_img_object: The moose_img_object to be resampled.
        :type moose_img_object: MooseImage
        :param interpolation: The interpolation method to use. Supported methods are 'nearest', 'linear', and 'bspline'.
        :type interpolation: str
        :param desired_spacing: The new spacing to use.
        :type desired_spacing: tuple
        :param desired_size: The new size to use. Default is None.
        :type desired_size: tuple
        :return: The resampled image as nibabel.Nifti1Image.
        :rtype: nibabel.Nifti1Image
        """

        image_header = moose_img_object.original_header
        image_affine = moose_img_object.image.affine
        sitk_input_image = moose_img_object.sitk_image
        # Resampling scheme based on image size
        if moose_img_object.is_large:
            resampled_sitk_image = ImageResampler.resample_image_SimpleITK_DASK(sitk_input_image, interpolation,
                                                                                desired_spacing, desired_size)
        else:
            resampled_sitk_image = ImageResampler.resample_image_SimpleITK(sitk_input_image, interpolation,
                                                                           desired_spacing, desired_size)

        new_size = resampled_sitk_image.GetSize()

        # Edit affine to fit the new image
        new_affine = image_affine
        for diagonal, spacing in enumerate(desired_spacing):
            new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
                new_affine[diagonal, diagonal])) * spacing

        # Edit header to fit the new image
        image_header['pixdim'][1:4] = desired_spacing
        image_header['dim'][1:4] = new_size
        image_header['srow_x'] = new_affine[0, :]
        image_header['srow_y'] = new_affine[1, :]
        image_header['srow_z'] = new_affine[2, :]

        resampled_image = nibabel.Nifti1Image(sitk.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                              affine=new_affine,
                                              header=image_header)

        return resampled_image

    @staticmethod
    def resample_segmentations(input_image_path: str, desired_spacing: tuple,
                               desired_size: tuple) -> nibabel.Nifti1Image:
        """
        Resamples an image to a new spacing.

        :param input_image_path: Path to the input image.
        :type input_image_path: str
        :param desired_spacing: The new spacing to use.
        :type desired_spacing: tuple
        :param desired_size: The new size to use.
        :type desired_size: tuple
        :return: The resampled image as nibabel.Nifti1Image.
        :rtype: nibabel.Nifti1Image
        """
        # Load the image and get necessary information
        input_image = nibabel.load(input_image_path)
        image_data = input_image.get_fdata()
        image_header = input_image.header
        image_affine = input_image.affine
        original_spacing = image_header.get_zooms()
        translation_vector = image_affine[:3, 3]
        rotation_matrix = image_affine[:3, :3]

        # Convert to SimpleITK image format
        image_data_swapped_axes = image_data.swapaxes(0, 2)
        sitk_input_image = sitk.GetImageFromArray(image_data_swapped_axes)
        sitk_input_image.SetSpacing([spacing.item() for spacing in original_spacing])
        axis_flip_matrix = np.diag([-1, -1, 1])
        sitk_input_image.SetOrigin(np.dot(axis_flip_matrix, translation_vector))
        sitk_input_image.SetDirection(
            (np.dot(axis_flip_matrix, rotation_matrix) / np.absolute(original_spacing)).ravel())

        desired_spacing = np.array(desired_spacing).astype(np.float64)

        # Interpolation:
        resampled_sitk_image = sitk.Resample(sitk_input_image, desired_size, sitk.Transform(),
                                             sitk.sitkNearestNeighbor,
                                             sitk_input_image.GetOrigin(), desired_spacing,
                                             sitk_input_image.GetDirection(), 0.0, sitk_input_image.GetPixelIDValue())

        # Edit affine to fit the new image
        new_affine = image_affine
        for diagonal, spacing in enumerate(desired_spacing):
            new_affine[diagonal, diagonal] = (new_affine[diagonal, diagonal] / abs(
                new_affine[diagonal, diagonal])) * spacing

        # Edit header to fit the new image
        image_header['pixdim'][1:4] = desired_spacing
        image_header['dim'][1:4] = desired_size
        image_header['srow_x'] = new_affine[0, :]
        image_header['srow_y'] = new_affine[1, :]
        image_header['srow_z'] = new_affine[2, :]

        resampled_image = nibabel.Nifti1Image(sitk.GetArrayFromImage(resampled_sitk_image).swapaxes(0, 2),
                                              affine=new_affine,
                                              header=image_header)

        return resampled_image

    @staticmethod
    def reslice_identity(reference_image: sitk.Image, moving_image: sitk.Image,
                         output_image_path: str = None, is_label_image: bool = False) -> sitk.Image:
        """
        Reslices an image to the same space as another image.

        :param reference_image: The reference image.
        :type reference_image: SimpleITK.Image
        :param moving_image: The image to reslice to the reference image.
        :type moving_image: SimpleITK.Image
        :param output_image_path: Path to the resliced image. Default is None.
        :type output_image_path: str
        :param is_label_image: Determines if the image is a label image. Default is False.
        :type is_label_image: bool
        :return: The resliced image as SimpleITK.Image.
        :rtype: SimpleITK.Image
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)

        if is_label_image:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)

        resampled_image = resampler.Execute(moving_image)
        resampled_image = sitk.Cast(resampled_image, sitk.sitkInt32)
        if output_image_path is not None:
            sitk.WriteImage(resampled_image, output_image_path)
        return resampled_image


def mip_3d(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Creates a Maximum Intensity Projection (MIP) of a 3D image.

    :param img: The input image.
    :type img: numpy.ndarray
    :param angle: The angle to rotate the image by.
    :type angle: float
    :return: The MIP of the rotated image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    # Rotate the image
    rot_img = rotate(img, angle, axes=(1, 2), reshape=False)

    # Create Maximum Intensity Projection along the first axis
    mip = np.max(rot_img, axis=1)

    # Invert the mip
    mip_inverted = np.max(mip) - mip

    # Rotate MIP 90 degrees anti-clockwise
    mip_flipped = np.flip(mip_inverted, axis=0)

    return mip_flipped


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to its maximum intensity.

    :param img: The input image.
    :type img: numpy.ndarray
    :return: The normalized image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    # Normalize the image to its maximum intensity
    img = img / np.max(img)

    return img


def equalize_hist(img: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to an image.

    :param img: The input image.
    :type img: numpy.ndarray
    :return: The equalized image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    img_eq = exposure.equalize_adapthist(img)

    return img_eq


def create_rotational_mip_gif(pet_path: str, mask_path: str, gif_path: str, rotation_step: int = 5,
                              output_spacing: Tuple[int, int, int] = (2, 2, 2)) -> None:
    """
    Creates a Maximum Intensity Projection (MIP) GIF of a PET image and its corresponding mask, rotating the image by a specified angle at each step.

    :param pet_path: The path to the PET image file.
    :type pet_path: str
    :param mask_path: The path to the mask image file.
    :type mask_path: str
    :param gif_path: The path to save the output GIF file.
    :type gif_path: str
    :param rotation_step: The angle to rotate the image by at each step.
    :type rotation_step: int
    :param output_spacing: The output voxel spacing of the resampled image.
    :type output_spacing: Tuple[int, int, int]
    :return: None
    :rtype: None
    """
    # Load the images
    sitk_pet_img = sitk.ReadImage(pet_path)
    sitk_mask_img = sitk.ReadImage(mask_path)

    mask_array = sitk.GetArrayFromImage(sitk_mask_img)
    if np.all(mask_array == 0):  # Check if the mask is empty
        logging.info(f"Warning: The mask at {mask_path} is empty. Processing PET image without mask overlay.")
        mask_overlay = False
    else:
        mask_overlay = True

    # Resample the images
    resampler = ImageResampler()
    sitk_pet_img_resampled = resampler.resample_image_SimpleITK_DASK(sitk_pet_img, 'linear', output_spacing)
    sitk_mask_img_resampled = resampler.resample_image_SimpleITK_DASK(sitk_mask_img, 'nearest', output_spacing)

    # Convert back to numpy array
    pet_img_resampled = sitk.GetArrayFromImage(sitk_pet_img_resampled)
    mask_img_resampled = sitk.GetArrayFromImage(sitk_mask_img_resampled) if mask_overlay else None

    # Normalize the PET image
    pet_img_resampled = normalize_img(pet_img_resampled)

    # Apply histogram equalization to PET image
    pet_img_resampled = equalize_hist(pet_img_resampled)

    # Create color versions of the images
    pet_img_color = np.stack((pet_img_resampled, pet_img_resampled, pet_img_resampled), axis=-1)  # RGB
    mask_img_color = np.stack((0.5 * mask_img_resampled, np.zeros_like(mask_img_resampled), 0.5 * mask_img_resampled),
                              axis=-1) if mask_overlay else None  # RGB, purple color

    # Create a Dask client with default settings
    client = Client()

    # Scatter the data to the workers
    pet_img_color_future = client.scatter(pet_img_color, broadcast=True)
    mask_img_color_future = client.scatter(mask_img_color, broadcast=True) if mask_overlay else None

    # Create MIPs for a range of angles and store them
    angles = list(range(0, 360, rotation_step))
    pet_mip_images_futures = client.map(mip_3d, [pet_img_color_future] * len(angles), angles)
    mask_mip_images_futures = client.map(mip_3d, [mask_img_color_future] * len(angles), angles) if mask_overlay else []

    # Gather the images
    pet_mip_images = client.gather(pet_mip_images_futures)
    mask_mip_images = client.gather(mask_mip_images_futures) if mask_overlay else []

    if mask_overlay:
        # Blend the PET and mask MIPs
        overlay_mip_images = [cv2.addWeighted(pet_mip, 0.7, mask_mip.astype(pet_mip.dtype), 0.3, 0)
                              for pet_mip, mask_mip in zip(pet_mip_images, mask_mip_images)]
    else:
        overlay_mip_images = pet_mip_images

    # Normalize the image array to 0-255
    mip_images = [(255 * (im - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8) for im in overlay_mip_images]

    # Save as gif
    imageio.mimsave(gif_path, mip_images, duration=FRAME_DURATION)

    # Cleanup
    client.close()
    del sitk_pet_img, sitk_mask_img, pet_img_resampled, mask_img_resampled, pet_img_color, mask_img_color, pet_mip_images, mask_mip_images, overlay_mip_images, mip_images


def compute_tumor_metrics(mask_path: str, pet_path: str):
    # Load images
    mask_img = sitk.ReadImage(mask_path)
    pet_img = sitk.ReadImage(pet_path)

    # Convert images to numpy arrays
    mask_array = sitk.GetArrayFromImage(mask_img)
    pet_array = sitk.GetArrayFromImage(pet_img)

    # Check if the mask is empty
    if np.all(mask_array == 0):
        logging.info(f"Warning: The mask at {mask_path} contains no tumor regions.")
        return 0, 0  # Return 0 for both tumor volume and average intensity

    # Compute voxel volume
    spacing = mask_img.GetSpacing()
    voxel_volume = np.prod(spacing)

    # Calculate tumor volume
    tumor_voxel_count = np.sum(mask_array == 1)  # assuming tumor is labeled with 1
    tumor_volume = (tumor_voxel_count * voxel_volume) / 1000  # convert to cm^3

    # Calculate average PET intensity within the tumor
    tumor_intensity_values = pet_array[mask_array == 1]
    average_intensity = np.mean(tumor_intensity_values)

    return tumor_volume, average_intensity


def save_metrics_to_csv(tumor_volume, avg_intensity, output_file):
    # Check if file exists to decide whether to write headers
    write_header = not os.path.exists(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['Tumor Volume (cm^3)', 'Average PET Intensity (Bq/ml)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({'Tumor Volume (cm^3)': tumor_volume, 'Average PET Intensity (Bq/ml)': avg_intensity})


def threshold_segmentation(pet_image_path:str, segmentation_data:np.array, intensity_threshold:int):
    """
    Thresholds the segmentation to only contain voxels with intensity higher than a specified value in the PET image.

    Parameters:
    pet_image_path (str): Path to the PET image NIfTI file.
    segmentation_path (str): Path to the input predicted segmentation NIfTI file.
    output_path (str): Path to save the thresholded segmentation NIfTI file.
    intensity_threshold (int): The intensity threshold. Voxels with PET intensity higher than this value will be kept.
    """
    # Load the PET image NIfTI file
    pet_img = nibabel.load(pet_image_path)
    pet_data = pet_img.get_fdata()

    # Apply the threshold from the PET image
    thresholded_data = np.logical_and(segmentation_data, pet_data > intensity_threshold).astype(np.uint8)

    return thresholded_data




