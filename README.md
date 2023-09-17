![Lion-logo.png](/Images/Lion-logo.png)
## LION (Lesion segmentatION): Loud. Proud. Unbounded. 🦁
[![Recommended Version](https://img.shields.io/badge/Recommended-pip%20install%20lionz%3D%3D0.2.0-9400D3.svg)](https://pypi.org/project/lionz/0.2.0/) 
[![Monthly Downloads](https://img.shields.io/pypi/dm/lionz?label=Downloads%20(Monthly)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/lionz/) 
[![Daily Downloads](https://img.shields.io/pypi/dd/lionz?label=Downloads%20(Daily)&color=9400D3&style=flat-square&logo=python)](https://pypi.org/project/lionz/)


LION has roared onto the scene, revolutionizing the way we view lesion segmentation. Born from the same lineage as MOOSE 2.0, LION is laser-focused on PET tumor segmentation. Our curated models, each crafted with precision, cater to various tracers, setting the gold standard in lesion detection.

✨ **Exclusive Engineering Paradigm**: With LION, segmentation is not just a task; it's an orchestrated dance of models. Define workflows for each model, mix and match channels as some models thrive on PET/CT while others are optimized for just PET. Run them in a sequence that maximizes output and efficiency. This unique trait of LION lets you tailor the process to your exact needs, making lesion segmentation an art of precision.

🌐 **Autopet-fueled Power**: LION has been trained on 1014 Autopet datasets. This vast and focused dataset empowers LION to spot and segment lesions with exceptional accuracy.

🔔 **Flexibility Unleashed**: Whether you're looking for a command-line tool for batch processing or a library package for your Python projects, LION has you covered. Seamlessly integrate it into your work environment and watch it shine.

Dive into the exciting world of PET tumor segmentation with LION and experience the future today!

---

## **Requirements** ✅

For an optimal experience with LION, ensure the following:

- **Operating System**: LION runs smoothly on Windows, Mac, or Linux.
- **Memory**: At least 32GB of RAM ensures LION operates without a hitch.
- **GPU**: For blazing-fast results, an NVIDIA GPU comes highly recommended. But if you don't have one, fret not! LION will still get the job done, just at a more leisurely pace.
- **Python**: Version 3.9.2 or above. We like to stay updated!

---

## **Installation Guide** 🛠️

Navigating the installation process is a breeze. Just follow the steps below:

**For Linux and MacOS** 🐧🍏
1. Create a Python environment, for example, 'lion-env'.
```bash
python3 -m venv lion-env
```
2. Activate your environment.
```bash
source lion-env/bin/activate  # for Linux
source lion-env/bin/activate  # for MacOS
```
3. Install LION.
```bash
pip install lionz
```

**For Windows** 🪟
1. Set up a Python environment, say 'lion-env'.
```bash
python -m venv lion-env
```
2. Get your environment up and running.
```bash
.\lion-env\Scripts\activate
```
3. Hop over to the PyTorch website and fetch the right version for your system. This step is crucial!
4. Finish up by installing LION.
```bash
pip install lionz
```

---

## **Usage Guide** 📚

**Command-line Tool for Batch Processing** 💻

Starting with LION is as intuitive as it gets. Here's how:

```bash
lionz -d <path_to_image_dir> -m <model_name>
```
Replace `<path_to_image_dir>` with your image directory and `<model_name>` with your chosen segmentation model's name.

And, if you ever find yourself needing some guidance:
```bash
lionz -h
```
This trusty command will spill all the beans about available models and their specialties.

**Using LION as a Library** 📚

Want to integrate LION in your Python code? Here's how:

1. Import the core function.
```python
from lionz import lion
```
2. Set up your variables and call `lion`.
```python
model_name = 'your_model_name'
input_dir = '/path_to_your_input'
output_dir = '/path_to_your_output'
accelerator = 'cuda_or_cpu'
lion(model_name, input_dir, output_dir, accelerator)
```

---

## **Directory Conventions for LION** 📂🏷️

For batch mode users ⚠️, ensure your data structure and naming conventions align with the provided guidelines. LION is compatible with both DICOM and NIFTI formats. For DICOM, LION discerns the modality from tags. For NIFTI, file naming is key.

**Directory Structure** 🌳

Organize your dataset as follows:

```
LION_data/
│
├── Subject1
│   ├── Modality1
│   │   └── File1.dcm
│   └── Modality2
│       └── File2.dcm
├── Subject2
│   └── Modality_Subject2.nii
└── Subject3
    └── Modality_Subject3.nii
```

## **Naming Conventions for NIFTI** 📝

Ensure you attach the correct modality as a prefix in the file name.

---

Harness the power of LION and elevate your PET tumor segmentation game

! 🚀🦁

Remember, the LION team is here to support you every step of the way. Should you need any assistance or if you'd like to provide feedback, don't hesitate to reach out to our dedicated support team in discord.

With LION by your side, your lesion segmentation adventures will be unstoppable! 😺🌟

Dive in now and make PET tumor segmentation a seamless experience!

---

Thank you for trusting LION with your PET tumor segmentation needs. We're committed to providing you with top-notch tools and services that make your work easier and more efficient.



---

## **A Note on QIMP Python Packages: The 'Z' Factor 📚🚀**

Every Python package at QIMP carries a unique mark – a distinctive 'Z' at the end of their names. This isn't just a quirk or a random choice. The 'Z' is emblematic, an insignia of our forward-leaning vision and unwavering dedication to continuous innovation.

Take, for instance, our LION package, dubbed 'lionz', pronounced "lion-zee". Now, one might wonder, why append a 'Z'?

In the expansive realm of science and mathematics, 'Z' is frequently invoked as a representation of the unexplored, the variables that are shrouded in mystery, or the ultimate point in a sequence. This mirrors our ethos at QIMP perfectly. We're inveterate boundary-pushers, ever eager to trek into the uncharted, always aligning ourselves with the vanguard of technological advancement. The 'Z' is a testament to this ethos. It symbolizes our ceaseless endeavor to transcend the conventional, to journey into the untouched, and to be the torchbearers of the future in medical imaging.

So, the next time you stumble upon a 'Z' in any of our package names, let it serve as a reminder of the zest for exploration and the spirit of discovery that fuels us. With QIMP, you're not merely downloading a tool; you're aligning yourself with a movement that aims to redefine the landscape of medical image processing. Let's soar into the realms of the 'Z' dimension, side by side! 🚀

---
