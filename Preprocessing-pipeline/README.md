# DeepForest: Sensing Into Self-Occluding Vegetation Volumes With Aerial Imaging

## Overview
**Abstract** Access to below-canopy volumetric vegetation data is crucial for understanding ecosystem dynamics, such as understory vegetation health, carbon sequestration, habitat structure, and biodiversity. We address a long-standing limitation of remote sensing, which often fails to penetrate below dense canopy layers. Thus far, LiDAR and radar are considered the primary options for measuring 3D vegetation structures, while cameras are only capable of extracting reflectance and depth of top layers. Our approach offers sensing deep into self-occluding vegetation volumes, such as forests, using conventional, high-resolution aerial images. It is similar, in spirit, to the imaging process of wide-field microscopy – yet handling much larger scales and strong occlusion. We scan focal stacks through synthetic aperture imaging with drones and remove out-of-focus contributions using pre-trained 3D convolutional neural networks. The resulting volumetric reflectance stacks contain low frequent representations of the vegetation volume. Combining multiple reflectance stacks from various spectral channels provides insights into plant health, growth, and environmental conditions throughout the entire vegetation volume.


## Features
- Pre-process multi-spectral vegetation datasets.
- Train deep learning models for vegetation analysis.
- Perform inference and corrections on processed vegetation data.

---

## Getting Started

### Prerequisites
Before using this project, ensure you have the following installed:
- Python 3.8 or later
- Required libraries (install using `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### Data Preprocessing
1. Place all the multi-spectral data in the `dataset` folder. Ensure each channel is stored in a separate subfolder within the `dataset` directory.
   - Example structure:
     ```
     dataset/
       channel1/
       channel2/
       channel3/
     ```

2. Navigate to the `preprocess` folder:
   ```bash
   cd preprocess
   ```

3. Run the preprocessing pipeline:
   ```bash
   python main.py
   ```

### Training
To train the model, run the following command from the project directory:
```bash
python train.py
```

### Inference
To perform inference or correction on new data, use:
```bash
python test.py
```
Ensure the pre-trained weights is stored in the same directory.

