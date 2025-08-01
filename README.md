# Mid-level feature representations 

This repository contains code for the paper "Investigating the temporal dynamics and modelling of mid-level feature representations in humans" (Karapetian et al., 2025). 

The data and stimulus set from the paper can be found at [https://osf.io/7c9bz/](https://osf.io/7c9bz/).

For questions about the code or the paper, please email agnessakarapetian@gmail.com or lxlenders@gmail.com. 

Please cite the paper if you use any of the code or data. 

## **EEG** 

This code is used to reproduce the results for Figures 4 and 5 of the paper, pertaining to the decoding and encoding analyses of the EEG data. 

### **1. Preprocessing** 

To preprocess the EEG data according to the steps described in the paper, use [preprocessing_eeg.py](EEG/Preprocessing/preprocessing_eeg.py)

### **2. Decoding**

**2.1 Analysis**

To obtain the decoding accuracy timecourses for image and video data, use [decoding.py](EEG/Decoding/decoding.py).

**2.2 Stats**

To calculate the significance statistics for the decoding analysis, use [decoding_significance_stats.py](EEG/Stats/decoding_significance_stats.py) (images and videos) and [decoding_difference_significance_stats.py](EEG/Stats/decoding_difference_significance_stats.py) (difference curve). To calculate the 95% confidence intervals for the decoding timecourses and peak latencies, use [decoding_bootstrapping.py](EEG/Stats/decoding_bootstrapping.py) (images and videos) and [decoding_difference_bootstrapping.py](EEG/Stats/decoding_difference_bootstrapping.py) (difference curve). 

**2.3 Plotting**

To plot the results and the stats of the decoding analysis, use [plot_decoding.py](EEG/Plotting/plot_decoding.py).

### 3. **Encoding**

**3.1 Preparation of ground-truth annotations**

To prepare the mid-level feature ground-truth annotations for the encoding analysis, as well as extract the annotations for low- and high-level features (canny edges and action identity), use [annotation_prep_images.py](EEG/Encoding/annotation_prep_images.py) (images) and [annotation_prep_videos.py](EEG/Encoding/annotation_prep_videos.py) (videos).

**3.2 MVNN**

Prior to encoding, we need to normalize the EEG data using MVNN (see Guggenmos et al., 2018). For this, use [mvnn_encoding.py](EEG/Encoding/mvnn_encoding.py).

**3.3 Hyperparameter optimization**
  
To optimize the hyperparameter lambda of the ridge-regression, use [hyperparameter_optimization.py](EEG/Encoding/hyperparameter_optimization.py). 

**3.4 Regression**

To perform the encoding analysis (ridge regression) using the optimized lambda values and obtain the EEG encoding time-courses, use [encoding.py](EEG/Encoding/encoding.py).

**3.5 Stats**

To calculate the EEG encoding significance statistics, use [encoding_significance_stats.py](EEG/Stats/encoding_significance_stats.py) (images and videos) and [encoding_difference_significance_stats.py](EEG/Stats/encoding_difference_significance_stats.py) (difference curve). To calculate the 95% confidence intervals for the encoding timecourses and peak latencies, use [encoding_bootstrapping.py](EEG/Stats/encoding_bootstrapping.py) (images and videos) and [encoding_difference_bootstrapping.py](EEG/Stats/encoding_difference_bootstrapping.py) (difference curve). 

**3.6 Plotting**

To plot the results and the stats of the EEG encoding analyses, use [plot_encoding.py](EEG/Plotting/plot_encoding.py)

## **CNN** 

This code is used to reproduce the results for the Figures 6 and 7, pertaining to the encoding analyses on the CNN activations and the EEG-CNN comparison.

### 1. Activation extraction and preparation

**1.1 Activation extraction**

To extract activations from ResNet-18 (He et al. 2015), a CNN trained on scene categorization with images (Zhou et al., 2018), use [activation_extraction_cnn_images.ipynb](CNN/Activation_extraction_and_prep/activation_extraction_cnn_images.ipynb). 
To extract activations from 3D-ResNet18 (Tran et al., 2018), a CNN trained on action recognition with videos (Kay et al., 2017), use [activation_extraction_cnn_videos.ipynb](CNN/Activation_extraction_and_prep/activation_extraction_cnn_videos.ipynb). 

**1.2 PCA**

To perform principal component analysis (PCA) on the extracted activations, use [pca_activations.py](CNN/Activation_extraction_and_prep/pca_activations.py). 

**1.3 Split activations**

To save activations separately in training/test/validation splits, use [prepare_layers.py](CNN/Activation_extraction_and_prep/prepare_layers.py). 

### 2. Encoding

**2.1 Hyperparameter optimization**

To optimize the hyperparameter lambda of the ridge-regression, use [hyperparameter_optimization_cnn.py](CNN/Encoding/hyperparameter_optimization_cnn.py).

**2.2 Ridge-regression**

To perform the encoding analysis (ridge regression) using the optimized lambda values and obtain the CNN encoding timecourses, use [encoding_cnn.py](CNN/Encoding/encoding_cnn.py).

**2.3 Stats**

To calculate the CNN encoding significance statistics, use [encoding_significance_stats_cnn.py](CNN/Stats/encoding_significance_stats_cnn.py) (images and videos) and [encoding_difference_significance_stats_cnn.py](CNN/Stats/encoding_difference_significance_stats_cnn.py) (difference curve). To calculate the 95% confidence intervals for the encoding timecourses and peak latencies, use [encoding_bootstrapping_cnn.py](CNN/Stats/encoding_bootstrapping_cnn.py) (images and videos) and [encoding_difference_bootstrapping_cnn.py](CNN/Stats/encoding_difference_bootstrapping_cnn.py) (difference curve). 

**2.4 Plotting**

To plot the results and stats of the CNN encoding analyses, use [encoding_plot_cnn.py](CNN/Plotting/encoding_plot_cnn.py).

To calculate and plot the correlation between EEG and CNN encoding results, use [eeg_vs_cnn_corr_peak_latencies.py](CNN/Plotting/eeg_vs_cnn_corr_peak_latencies.py).

### 3. Folder structure
Created with: tree -L 3  -d /scratch/alexandel91/mid_level_features > ./directory_structure.txt
```
├── data
│   ├── CNN
│   │   ├── 2dresnet18_pretrained
│   │   └── 3dresnet18
│   └── EEG
│       ├── images
│       └── miniclips
├── features
│   ├── images
│   │   └── default
│   └── miniclips
│       ├── control_1
│       ├── control_2
│       └── default
├── kinetics_400
│   ├── train
│   │   ├── annotations
│   │   ├── files
│   │   ├── tars
│   │   └── train
│   └── val
│       ├── annotations
│       ├── files
│       ├── tars
│       └── val
├── results
│   ├── c4
│   ├── c5
│   ├── c7_1
│   ├── c7_2
│   ├── c7_3
│   ├── CNN
│   │   ├── control_10
│   │   ├── control_11
│   │   ├── control_12
│   │   ├── default
│   │   └── training
│   ├── EEG
│   │   ├── control_1
│   │   ├── control_12
│   │   ├── control_2
│   │   ├── control_3
│   │   ├── control_6_1
│   │   ├── control_6_2
│   │   ├── control_9
│   │   └── default
│   └── noise_ceiling
│       ├── images
│       └── miniclips
└── stimuli
    ├── images
    │   ├── frame_annotations
    │   └── frames
    └── miniclips
        ├── frame_annotations
        ├── frames
        └── mp4

57 directories
```

