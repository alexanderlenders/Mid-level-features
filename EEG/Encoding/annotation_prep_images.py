#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANNOTATION PREPARATION AND PCA - IMAGES - UNREAL ENGINE

This script prepares the annotations from the Unreal Engine by extracting the low-, mid-, and 
high-level features from them. For the low-level feature, canny edges, the canny algorithm is applied.
Lastly, a PCA is performed on all features which have more than n=100 components. 

Anaconda-environment on local machine: opencv_env

@author: AlexanderLenders
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------
import argparse

# parser
parser = argparse.ArgumentParser()

# add arguments / inputs
parser.add_argument('-c', "--n_components", default=100, type=int, metavar='', 
                    help="number of components for PCA")
parser.add_argument('-pca', "--pca_method", default='linear', type=str, metavar='', 
                    help="linear PCA or nonlinear Kernel PCA")
parser.add_argument('-id', "--imgdir", 
                    default = 'Z:/Unreal/single_frames',
                    type = str, metavar='', help="Image directory; update as needed")
parser.add_argument('-ad', "--annodir", 
                    default = 'Z:/Unreal/single_frames_annotations',
                    type = str, metavar='', help="Annotations directory; update as needed")
parser.add_argument('-actd', "--actiondir", 
                    default = 'Z:/Unreal/Results/features/action_indices.csv',
                    type = str, metavar='', help="Action metadata directory; update as needed")
parser.add_argument('-chard', "--charactdir", 
                    default = 'Z:/Unreal/Results/features/meta_data_anim.mat',
                    type = str, metavar='', help="Character metadata directory; update as needed")
parser.add_argument('-sd', "--savedir", 
                    default = 'Z:/Unreal/images_results/encoding',
                    type = str, metavar='', help="Save results directory; update as needed")
parser.add_argument('-f', "--frame", 
                    default = 20, type = int, metavar='', help="frame number")
args = parser.parse_args() # to get values for the arguments

n_components = args.n_components
pca_method = args.pca_method
images_dir = args.imgdir
annotations_dir = args.annodir
action_dir = args.actiondir
character_dir = args.charactdir
save_dir = args.savedir
frame = args.frame

# -----------------------------------------------------------------------------
# STEP 2: Define function
# -----------------------------------------------------------------------------

def feature_extraction(n_components, annotations_dir, character_dir, action_dir, save_dir, pca_method, frame): 
    """
    Standard Scaler and PCA are fitted only on the training data and applied
    to the training, test and validation data. By default this function makes
    use of the canny edge extraction function in the openCV module.
    
    Input: 
    ----------
    a. Single frame (frame 20)
    b. Single frame annotations from Unreal Engine 
    c. Metadata containing information about data split and character/action
    
    Returns
    ----------
    image_features.pkl: Canny edges, World normals, Lighting, Scene Depth, 
    Reflectance, Character Identity, Action Identity, Skeleton Position after
    PCA (if necessary), saved in a dictionary "image_features"
        - Dictionary contains matrix for each feature with the dimension 
        num_images x num_components 
    
    Parameters
    ----------    
    n_components: int 
        Number of components for the PCA
    annotations_dir: str 
        Directory with single frame annotations as .jpg and .pkl
    character_dir: str
        Directory with character information (for train/test/val split)
    action_dir: str
        Directory with action identity information
    save_dir: str
        Directory where to save the output of the function 
    pca_method: str 
        Whether to use a linearPCA ('linear') or KernelPCA ('nonlinear')
    frame: int 
        Image frame where to get annotations
    """
    
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules, Define Variables, Load Meta Data
    # -------------------------------------------------------------------------
    
    # Import modules 
    import os
    import pickle
    import numpy as np 
    import cv2 as cv 
    from scipy.ndimage import convolve, gaussian_filter 
    import scipy.io    
    from PIL import Image
    import pandas as pd 
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler
    
    # Feature names 
    feature_names = ('edges','skeleton', 'world_normal','lightning','scene_depth', 'reflectance','action') #CHANGE LIGHTNING TO LIGHTING
    
    # Number of images 
    num_images = 1440 
    
    # Load meta data for character identity - for the train/test/val split
    ## Character Identity
    meta_data = scipy.io.loadmat(character_dir)
    char_data = pd.DataFrame((meta_data.get("meta_data")))
    rows, cols = char_data.shape
    char_meta = pd.DataFrame(np.zeros((rows, cols))) 

    for col in range(cols):
        extracted_values = [item.item() for item in char_data.iloc[:, col]]
        char_meta[char_meta.columns[col]] = extracted_values
    
    ## Action identity
    action_data = pd.read_csv(action_dir, header = None)

    # Recode the six different actions
    actions = {1: 0, 2: 1, 9: 2, 18: 3, 19: 4, 30: 5}
    action_data[1] = action_data[1].replace(actions)

    # Get information about train and test data
    split_data = np.array(char_meta.iloc[:, 1])
    # 0 -> training data
    # 1 -> validation data
    # 2 -> test data
    index = np.arange(0,num_images)
    split_data = np.column_stack([index, split_data])
    train_data = split_data[:, 0][split_data[:, 1] == 0]
    val_data = split_data[:, 0][split_data[:, 1] == 1]
    test_data = split_data[:, 0][split_data[:, 1] == 2]

    # -------------------------------------------------------------------------
    # STEP 2.2 Canny Edge Detection
    # -------------------------------------------------------------------------
    """
    There are different possibilities to extract edges from the image frames. 
    First, note that we will use only a single frames. 
    Second, there is a variety of gradient operators for detecting edges.
    For more information, see: 
        https://cave.cs.columbia.edu/Statics/monographs/Edge%20Detection%20FPCV-2-1.pdf
        https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
        https://masters.donntu.ru/2010/fknt/chudovskaja/library/article5.htm
    For consistency with the image paradigm, we will do a Canny edge detection 
    (see above, for more information) with a Sobbel 3x3 operator. 
    Regarding the thresholds for non-maximum surpression, see: 
        https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
    """
    
    def canny_edge(image, images_dir, frame, openCV = True, gaussian_filter_size = 3):      
        """        
        Parameters
        ----------
        image: int 
            Number of images 
        images_dir: str 
            Directory of images
        frame: int 
            Image frame
        openCV: bool 
            Use openCV canny edge detection function 
        gaussian_filter_size: int 
            Size of the gaussian filter to reduce noise in the image 
        """
        # Import image 
        image_file = str(image).zfill(4) + '_frame_{}'.format(frame) + '.jpg' # zfill: fill with zeros (4)
        image_dir = images_dir + '/' +  image_file
        
        if openCV is True: 
            img = cv.imread(image_dir)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # It is recommended to blur before doing canny edge detection
            blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
            
            # Recommended thresholds in a stack overflow question
            # high_threshold = 255
            # low_threshold = high_threshold/3
            
            # Values as in YouTube Tutorial:
            canny_edges = cv.Canny(blur, 125, 175)

        elif openCV is False: 
            img = Image.open(image_dir)
            img = np.array(img.convert('L')).astype(np.float32)
            # L means we convert it to a grey-valued image
            
            # Gaussian blur to reduce noise level
            gaussian_image = gaussian_filter(img, gaussian_filter_size)
            # We could also try out a 5x5x5 filter (as recommended in the openCV 
            # tutorial)
        
            # Use sobel filters to get gradients with respect to x and y
            grad_x = convolve(gaussian_image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            grad_y = convolve(gaussian_image, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
            # Compute the magnitude of the gradient (:= edge strength)
            canny_edges = np.power(np.power(grad_x, 2.0) + np.power(grad_y, 2.0), 0.5)
            
            # Compute the edge direction
            # theta = np.arctan2(grad_y, grad_x)
            
            # What is missing here in comparison to the openCV implementation is 
            # the non-maximum surpression and hysteresis thresholding
        
        return canny_edges


    # -------------------------------------------------------------------------
    # STEP 2.3 Action Identity
    # -------------------------------------------------------------------------
    def action(image):
        """        
        Parameters
        ----------
        image: int 
            Number of image
        """
        action_id = action_data.iloc[image, 1]
        one_hot_vector = np.zeros([len(actions),])
        one_hot_vector[action_id] = 1
        return one_hot_vector


    # -------------------------------------------------------------------------
    # STEP 2.4 Skeleton Position
    # -------------------------------------------------------------------------
    def skeleton_pos(image, annotations_dir, frame):
        """        
        Parameters
        ----------
        image: int 
            Number of image 
        annotations_dir: str
            Directory with single frame annotations as .pkl
        frame: int 
            Image frame
        """
        # Import image 
        image_file = str(image).zfill(4) + '_skeleton_position_frame_{}'.format(frame) + '.pkl' # zfill: fill with zeros (4)
        image_dir = annotations_dir + '/' + image_file
        pickle = np.load(image_dir, allow_pickle=True)
        # Extract bone names
        # bone_names = pickle['bone_name'].tolist()
        position_x_y = np.array(pickle[['screen_pos_x', 'screen_pos_y']])
        return position_x_y


    # -------------------------------------------------------------------------
    # STEP 2.5 World Normals
    # -------------------------------------------------------------------------
    def world_normals(image, annotations_dir, frame):
        """        
        Parameters
        ----------
        image: int 
            Number of image
        annotations_dir: str
            Directory with single frame annotations as .jpg
        frame: 
            Image frame
        """
        image_file = str(image).zfill(4) + '_world_normal' + '_frame_{}'.format(frame) + '.jpg'
        image_dir = annotations_dir + '/' + image_file 
        image = Image.open(image_dir)
        world_normals_rgb = np.array(image.convert('RGB')).astype(np.float32)
        return world_normals_rgb
        
    # -------------------------------------------------------------------------
    # STEP 2.6 Lighting
    # -------------------------------------------------------------------------
    def lighting(image, annotations_dir, frame):
        """        
        Parameters
        ----------
        image: int 
            Number of image 
        annotations_dir: str
            Directory with single frame annotations as .jpg
        frame: int 
            Image frame
        """
        image_file = str(image).zfill(4) + '_lightning' + '_frame_{}'.format(frame) + '.jpg'
        image_dir = annotations_dir + '/' + image_file 
        image = Image.open(image_dir)
        lightning_np = np.array(image.convert('L')).astype(np.float32)
        return lightning_np
            
    # -------------------------------------------------------------------------
    # STEP 2.7 Scene depth
    # -------------------------------------------------------------------------
    def depth(image, annotations_dir, frame):
        """        
        Parameters
        ----------
        image: int 
            Number of image 
        annotations_dir: str
            Directory with single frame annotations as .jpg
        frame: int 
            Image frame
        """
        image_file = str(image).zfill(4) + '_scene_depth' + '_frame_{}'.format(frame) + '.jpg'
        image_dir = annotations_dir + '/' + image_file 
        image = Image.open(image_dir)
        scene_depth = np.array(image.convert('L')).astype(np.float32)
        return scene_depth
                
    # -------------------------------------------------------------------------
    # STEP 2.8 Reflectance
    # -------------------------------------------------------------------------
    def reflectance(image, annotations_dir, frame):
        """        
        Parameters
        ----------
        image: int 
            Number of image
        annotations_dir: str
            Directory with single frame annotations as .jpg
        frame: int 
            Image frame
        """
        image_file = str(image).zfill(4) + '_reflectance' + '_frame_{}'.format(frame) + '.jpg'
        image_dir = annotations_dir + '/' + image_file 
        image = Image.open(image_dir)
        reflectance_rgb = np.array(image.convert('RGB')).astype(np.float32)
        return reflectance_rgb
    
    # -------------------------------------------------------------------------
    # STEP 2.9 PCA
    # -------------------------------------------------------------------------
    def pca(features_train, features_val, features_test, pca_method = pca_method,
            n_comp = n_components): 
        """        
        NOTE: This implements a (simple) PCA using SVD. One could also implement
        a multilinear PCA for the images with RGB channels.
            
        Parameters
        ----------
        features_train: numpy array 
            Matrix with dimensions num_images x num_components 
            In the case of RGB channels one first has to flatten the matrix
        features_test: numpy array 
            Matrix with dimensions num_images x num_components 
            In the case of RGB channels one first has to flatten the matrix
        features_val: numpy array 
            Matrix with dimensions num_images x num_components 
            In the case of RGB channels one first has to flatten the matrix
        pca_method: str 
            Whether to apply a "linear" or "nonlinear" Kernel PCA
        n_comp: int
            Number of fitted components in the PCA
        """
    
        # Standard Scaler (Best practice, see notes)
        scaler = StandardScaler().fit(features_train)
        scaled_train = scaler.transform(features_train)
        scaled_test = scaler.transform(features_test)
        scaled_val = scaler.transform(features_val)
        
        # Fit PCA on train_data
        if pca_method == 'linear':
            pca_image = PCA(n_components = n_comp, random_state = 42)
        elif pca_method == 'nonlinear': 
            pca_image = KernelPCA(n_components = n_comp, kernel = 'poly', 
                                  degree = 4, random_state = 42)
            
        pca_image.fit(scaled_train)
        
        if pca_method == 'linear':
            # Get explained variance
            per_var = np.round(pca_image.explained_variance_ratio_* 100, decimals=1)
            labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        
            explained_variance = np.sum(per_var)
              
        # Transform data 
        pca_train = pca_image.transform(scaled_train)
        pca_test = pca_image.transform(scaled_test)
        pca_val = pca_image.transform(scaled_val)

        return pca_train, pca_val, pca_test
        
    # -------------------------------------------------------------------------
    # STEP 2.10 Get features for all images and apply PCA
    # -------------------------------------------------------------------------
    pca_features = dict.fromkeys(feature_names)
    
    for feature in pca_features.keys():
        print(feature)
        
        datasets = []
        
        if feature == 'edges':
            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype = float)
            
            for img in range(num_images): 
                # features_flattened_frame = np.zeros((1, 202800), dtype = float)
                img_index = img + 1
                
                # for i, frame in enumerate(range(10, 18)): 
                feature_np = canny_edge(img_index, images_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img,:] = feature_flatten
            
            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            # PCA 
            pca_features_train, pca_features_val, pca_features_test = pca(features_train, 
                                                                          features_val, 
                                                                          features_test)
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)
        
        elif feature == 'skeleton': 
            # 14 skeleton positions * 2 coordinates
            features_flattened = np.zeros((num_images, 28), dtype = float)
            
            for img in range(num_images): 
                img_index = img + 1
                
                feature_np = skeleton_pos(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

            # Split data 
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            datasets.append(features_train)
            datasets.append(features_val)
            datasets.append(features_test)
        
        elif feature == 'world_normal': 
            
            # for RGB 390*520*3, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 608400), dtype = float)
            
            for img in range(num_images): 
                img_index = img + 1
                
                feature_np = world_normals(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten
            
            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            # PCA 
            pca_features_train, pca_features_val, pca_features_test = pca(features_train, 
                                                                          features_val, 
                                                                          features_test)
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)
            
        
        elif feature == 'lightning':
            
            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype = float)
            
            for img in range(num_images): 
                
                img_index = img + 1

                feature_np = lighting(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten

            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            # PCA 
            pca_features_train, pca_features_val, pca_features_test = pca(features_train, 
                                                                          features_val, 
                                                                          features_test)
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)
        
        elif feature == 'scene_depth':
            
            # for GRAY 390*520, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 202800), dtype = float)
            
            for img in range(num_images): 
                
                img_index = img + 1

                feature_np = depth(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()               
                features_flattened[img, :] = feature_flatten
            
            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            # PCA 
            pca_features_train, pca_features_val, pca_features_test = pca(features_train, 
                                                                          features_val, 
                                                                          features_test)
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)
        
        
        elif feature == 'reflectance':
            
            # for RGB 390*520*3, where 390*520 dimension of image
            features_flattened = np.zeros((num_images, 608400), dtype = float)
            
            for img in range(num_images): 
                img_index = img + 1
                
                feature_np = reflectance(img_index, annotations_dir, frame)
                feature_flatten = feature_np.flatten()
                features_flattened[img, :] = feature_flatten
            
            # Split data
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            # PCA 
            pca_features_train, pca_features_val, pca_features_test  = pca(features_train, 
                                                                          features_val, 
                                                                          features_test)
            del features_train, features_val, features_test
            datasets.append(pca_features_train)
            datasets.append(pca_features_val)
            datasets.append(pca_features_test)
        
        elif feature == 'action':
            features_flattened = np.zeros((num_images, len(actions)), dtype = float)
            
            for img in range(num_images): 
                feature_np = action(img)
                features_flattened[img, :] = feature_np
            
            # Split data 
            features_train = features_flattened[train_data]
            features_val = features_flattened[val_data]
            features_test = features_flattened[test_data]
            del features_flattened
            
            datasets.append(features_train)
            datasets.append(features_val)
            datasets.append(features_test)

        pca_features[feature] = datasets
        
    # -------------------------------------------------------------------------
    # STEP 2.11 Save Output
    # -------------------------------------------------------------------------
    features_dir = save_dir + '/' + 'img_features_frame_20_redone_7features_onehot.pkl'
    
    with open(features_dir, 'wb') as f:
        pickle.dump(pca_features, f)

# -----------------------------------------------------------------------------
# STEP 3: Run Function
# -----------------------------------------------------------------------------
feature_extraction(n_components, annotations_dir, character_dir, action_dir, save_dir, pca_method, frame)
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    