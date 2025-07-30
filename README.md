# DeepForest: Sensing Into Self-Occluding Volumes of Vegetation With Aerial Imaging

This repository contains the official authors implementation associated with the paper "DeepForest: Sensing Into Self-Occluding Volumes of Vegetation With Aerial Imaging", which can be found [paper link](https://arxiv.org/pdf/2502.02171).

![plot](./imgs/Fig_9.jpg)


Access to below-canopy volumetric vegetation data is crucial for understanding ecosystem dynamics. We address the long-standing limitation of remote sensing to penetrate deep into dense canopy layers. LiDAR and radar are currently considered the primary options for measuring 3D vegetation structures, while cameras can only extract the reflectance and depth of top layers. Using conventional, high-resolution aerial images, our approach allows sensing deep into self-occluding vegetation volumes, such as forests. It is similar in spirit to the imaging process of wide-field microscopy, but can handle much larger scales and strong occlusion. We scan focal stacks by synthetic-aperture imaging with drones and reduce out-of-focus signal contributions using pre-trained 3D convolutional neural networks. The resulting volumetric reflectance stacks contain low-frequency representations of the vegetation volume. Combining multiple reflectance stacks from various spectral channels provides insights into plant health, growth, and environmental conditions throughout the entire vegetation volume.


## Funding and Acknowledgments
- Linz Institute of Technology grant LIT-2022-11-SEE-112 (OB)
- Austrian Science Fund (FWF), German Research Foundation (DFG) grant I 6046-N (OB)

## How to Use

1. **Acquire Data**  
   Capture the raw data from the camera system.

2. **Preprocess the Data**  
   Run the preprocessing pipeline on all channels to clean and standardize the input data.

3. **Generate Integral Images**  
   Use the [AOS repository](https://github.com/JKU-ICG/AOS/) to generate integral images needed for further processing.  
   
4. **Prepare Training Data**  
   Use `utils.py` to generate the Point Spread Function (PSF) data required for training.

5. **Train the Model**  
   Run `train.py` to start training the model using the generated PSF data.

6. **Perform Inference**  
   Use `test.py` to evaluate the trained model on new or test data.

7. **Sensor Mapping**  
   Run `match_sensor.py` to align the corrected reflectance stack with the center perspective sensor image, as described in the paper.


## Cloning the Repository
```
>> git clone https://github.com/mohamedhaiham94/DeepForest-Sensing-Into-Self-Occluding-Volumes-of-Vegetation-With-Aerial-Imaging.git
```

## Setup
Set up conda enviroment and install packages

```
>> conda create -n deepforest python=3.11
>> conda activate deepforest
>> pip -r install requirements.txt
```

To install all required packages you need to run the following command

```
>> pip -r install requirements.txt
```

## Testing
To test our approach you have to download the checkpoints from the following [link](https://drive.google.com/file/d/1EzkNiE4O8C0CiEvDKRc8aXZqSPuWEPWj/view?usp=drive_link), then run the following code.

```
>> python test.py
```

To evaluate the model on new datasets, apply the identical preprocessing pipeline described in our paper. To generate the receptive fields (model inputs) run this script
```
>> python utils.py
```


