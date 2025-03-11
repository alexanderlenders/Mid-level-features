#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENCODING - AVERAGE FEATURES BEFORE PCA - UNREAL ENGINE

This script implements the multivariate linear ridge regression for the 
scene features from the Unreal Engine for a single frame. 

@author: AlexanderLenders
"""
# -----------------------------------------------------------------------------
# STEP 1: Initialize variables
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # add arguments / inputs
    parser.add_argument('-s', "--sub", default=9, type=int, metavar='', 
                        help="subject ID (see range below)")
    parser.add_argument('-f', "--freq", default=50,type=int, metavar='', 
                        help="downsampling frequency")
    parser.add_argument('-r', "--region", default="posterior",type=str, 
                        metavar='',
                        help="Electrodes to be included, posterior (19) or wholebrain (64)")
    parser.add_argument('-i', "--input_type", default = 'images', type = str, 
                        metavar='', help="Font")

    args = parser.parse_args() # to get values for the arguments
    
    sub = args.sub
    freq = args.freq
    region = args.region        
    input_type = args.input_type

# -----------------------------------------------------------------------------
# STEP 2: Define Encoding Function
# -----------------------------------------------------------------------------

def encoding(sub, freq, region, input_type): 
    """
    Input: 
    ----------
    I. Test, Training and Validation EEG data sets, which are already 
    preprocessed + MVNN. The input are dictionaries, which include: 
        a. EEG-Data (eeg_data, 5400 Images/Videos x 64 Channels x 70 Timepoints)
        b. Image/Video Categories (img_cat, 5400 x 1) - Each image/video has one specific ID
        c. Channel names (channels, 64 x 1 OR 19 x 1)
        d. Time (time, 70 x 1) - Downsampled timepoints of an image/video
        In case of the validation data set there are 900 images/videos instead of 5400.
    II. Image/Video features 
        a. image_features or video_features.pkl: Canny edges, World normals, Lighting, Scene Depth, 
        Reflectance, Action Identity, Skeleton Position after
        PCA (if necessary), saved in a dictionary "image_features" or "video_features"
            - Dictionary contains matrix for each feature with the dimension 
            num_videos x num_components 
        b. exp_variance_pca.pkl: Explained variance for each feature after PCA
        with n_components.
    III. Hyperparameter
    regression_features, dictionary with the following outputs for each feature: 
        a. Root mean square error (RMSE) matrix (Timepoints x Alpha] - rmse_score
        b. Pearson correlation between true EEG data and predicted EEG data - correlation
        c. Best alpha for each timepoint based on RMSE - best_alpha_rmse
        d. Best alpha for each timepoint based on correlation - best_alpha_corr
        e. Best alpha averaged over timepoints based on RMSE - best_alpha_a_rmse 
        f. Best alpha averaged over timepoints based on correlation - best_alpha_a_corr
    
    Returns - ADAPT ...
    ----------
    ...
    
    Parameters
    ----------
    sub : int 
        Subject number 
    freq : int
          Downsampling frequency (default is 50)
    region : str
        The region for which the EEG data should be analyzed. 
    input_type: str
        Miniclips or images

    """
    # -------------------------------------------------------------------------
    # STEP 2.1 Import Modules & Define Variables
    # -------------------------------------------------------------------------
    # Import modules
    import os
    import numpy as np
    import torch 
    import pickle
      
    feature_names = ('edges','world_normal','lightning','scene_depth','reflectance','skeleton','action')
    featuresDir = f'/home/agnek95/Encoding-midlevel-features/Results/Encoding/{input_type}/7_features/img_features_frame_20_redone_7features_onehot.pkl'
    
    features_dict = dict.fromkeys(feature_names)
    
    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Number of channels 
    if region == 'wholebrain': 
        n_channels = 64
    elif region == 'posterior': 
        n_channels = 19
           
    # -------------------------------------------------------------------------
    # STEP 2.2 Define Loading EEG Data Function
    # -------------------------------------------------------------------------
    
    def load_eeg(sub, img_type, region, freq, input_type):

        # Define the directory
        workDirFull = '/scratch/agnek95/Unreal/'

        #load mvnn data
        if input_type=='miniclips':
            if sub < 10:  
                folderDir = os.path.join(workDirFull, '{}_data'.format(input_type) + '/sub-0{}'.format(sub) + 
                                            '/eeg/preprocessing/ica' + '/' + img_type + '/' +
                                            region + '/')
                fileDir = (('sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy'.format(sub,img_type,freq,region)))

            else: 
                folderDir = os.path.join(workDirFull, '{}_data'.format(input_type) + '/sub-{}'.format(sub) + 
                                            '/eeg/preprocessing/ica' + '/' + img_type + '/' +
                                            region + '/')
                fileDir = (('sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy'.format(sub,img_type,freq,region)))
                
        elif input_type=='images':
            if sub < 10: 
                folderDir = os.path.join(workDirFull, '{}_data_prepared'.format(input_type) + '/prepared' + '/sub-0{}'.format(sub) + 
                                        '/{}/{}/{}hz/'.format(img_type,region,freq))
                fileDir = (('sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy'.format(sub,img_type,freq,region)))
            else:
                folderDir = os.path.join(workDirFull, '{}_data_prepared'.format(input_type) + '/prepared' + '/sub-{}'.format(sub) + 
                                        '/{}/{}/{}hz/'.format(img_type,region,freq))         
                fileDir = (('sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy'.format(sub,img_type,freq,region)))

        total_dir = os.path.join(folderDir, fileDir)
                
        # Load EEG data 
        data = np.load(total_dir, allow_pickle=True).item()

        eeg_data = data["eeg_data"]
        img_cat = data["img_cat"]
        
        del data 
        
        #Average over trials    
        if input_type == 'miniclips': 
            n_conditions = len(np.unique(img_cat))
            _, n_channels, timepoints = eeg_data.shape
            n_trials = img_cat.shape[0]
            n_rep = round(n_trials/n_conditions)
            
            y_prep = np.zeros((n_conditions, n_rep, n_channels, timepoints), dtype = float)
            
            for condition in range(n_conditions): 
                idx = np.where(img_cat == np.unique(img_cat)[condition])
                y_prep[condition, :, :, :] = eeg_data[idx, :, :]
        elif input_type == 'images': 
            _, n_channels, timepoints = eeg_data.shape
            if img_type == 'train':
                n_conditions = 1080
                n_rep = 5
            elif img_type == 'test':
                n_conditions = 180
                n_rep = 30
            elif img_type == 'val':
                n_conditions = 180
                n_rep = 5
            y_prep = eeg_data.reshape(n_conditions,n_rep,n_channels,timepoints)
           
        y = np.mean(y_prep, axis = 1)
        
        return y, timepoints
    
    # -------------------------------------------------------------------------
    # STEP 2.3 Define load features function
    # -------------------------------------------------------------------------
    def load_features(feature, featuresDir): 
        
        features = np.load(featuresDir, allow_pickle=True)
        X_prep = features[feature]
        
        X_train = X_prep[0]
        X_val = X_prep[1]
        X_test = X_prep[2]
        
        return X_train, X_val, X_test
    
    # -------------------------------------------------------------------------
    # STEP 2.4 Define load alpha function
    # -------------------------------------------------------------------------
    def load_alpha(sub, freq, region, feature, input_type, tp = 0, averaged = True, 
                   correlation = True): 
        # averaged == True, if alpha averaged over all timepoints is required
        # correlation == True, if alpha should be determined based on correlation
        # instead of RMSE

        savedDir = f'/home/agnek95/Encoding-midlevel-features/Results/Encoding/{input_type}/7_features/'
     
        fileDir = (str(sub) + '_seq_' + str(freq) + 'hz_' + region +
                    '_hyperparameter_tuning_averaged_frame_before_mvnn_7features_onehot' + '.pkl')  
     
        alphaDir = os.path.join(savedDir, fileDir)

        alpha_values = np.load(alphaDir, allow_pickle=True)
        
        if averaged is True: 
            if correlation is True:
                alpha = alpha_values[feature]['best_alpha_a_corr']
            else: 
                alpha = regression_features[feature]['best_alpha_a_rmse']
        else: 
            if correlation is True:
                alpha = regression_features[feature]['best_alpha_corr'][tp]
            else: 
                alpha = regression_features[feature]['best_alpha_rmse'][tp]
        
        return alpha
    
    # -------------------------------------------------------------------------
    # STEP 2.5 Define model class
    # -------------------------------------------------------------------------
    class OLS_pytorch(object):
        def __init__(self, use_gpu=False, intercept = True, ridge = True, alpha = 0):
            self.coefficients = []
            self.use_gpu = use_gpu
            self.intercept = intercept
            self.ridge = ridge
            self.alpha = alpha # penalty (alpha or lambda)

        def fit(self,X,y, solver = 'cholesky'): 
            """
            Details (Statistical approach): 
                https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/LinearRegression.pdf
            For skeleton position, we have to use the cholesky solver since the
            Hermetian matrix is not positive definit for the skeleton position.
            There are different solvers for ridge regression, each of them
            have their advantages.
            - Choleksy decomposition 
            - LU decomposition 
            - and so on: 
                https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
            - 
            """
            if len(X.shape) == 1:
                X = self.reshape_x(X)
            if len(y.shape) == 1:
                y = self.reshape_x(y)
            
            if (self.intercept) is True:
                X = self.concatenate_ones(X)

            # convert numpy array into torch
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
                    
            # if we use a gpu, we have to transfer the torch to it
            if (self.use_gpu) is True:
                X = X.cuda()
                y = y.cuda()
            
            if (self.ridge) is True: 
                rows, columns = X.shape
                _ , columns_y = y.shape
                
                # we use the data augmentation approach to solve the ridge
                # regression via OLS 
                
                penalty_matrix = np.eye((columns))
                penalty_matrix = torch.from_numpy(penalty_matrix * np.sqrt(self.alpha)).float()
                
                zero_matrix = torch.from_numpy(np.zeros((columns, columns_y))).float()
                
                if (self.use_gpu) is True: 
                    penalty_matrix = penalty_matrix.cuda()
                    zero_matrix = zero_matrix.cuda()
                
                X = torch.vstack((X, penalty_matrix))
                y = torch.vstack((y, zero_matrix))
                
                # Creates Hermitian positive-definite matrix
                XtX = torch.matmul(X.t(),X)
                
                Xty = torch.matmul(X.t(), y)
                
                # Solve it
                if solver == 'cholesky': 
                    # Choleksy decomposition, creates the lower triangle matrix
                    L = torch.linalg.cholesky(XtX)
                    
                    betas_cholesky = torch.cholesky_solve(Xty, L)
        
                    self.coefficients = betas_cholesky
                    return betas_cholesky
            
                elif solver == 'lstsq':
                    lstsq_coefficients, _, _, _ = torch.linalg.lstsq(Xty, XtX, rcond = None)
                    return lstsq_coefficients.t()
                    self.coefficients = lstsq_coefficients.t()
                
                elif solver == 'solve':
                    solve_coefficients = torch.linalg.solve(XtX, Xty)
                    self.coefficients = solve_coefficients


        def predict(self, entry):
            # entry refers to the features of the test data 
            entry = self.concatenate_ones(entry)
            entry = torch.from_numpy(entry).float()
            
            if (self.use_gpu) is True:
                entry = entry.cuda()
            prediction = torch.matmul(entry,self.coefficients)
            prediction = prediction.cpu().numpy()
            prediction = np.squeeze(prediction)
            return prediction

        def score(self, entry, y, channelwise = True):
            # This computes the root mean square error
            # We could compare this model score to the correlation 
            # We could also use the determination criterion (R^2)
            # The old code was changed, did not which score was computed.
            
            entry = self.concatenate_ones(entry)
            
            entry = torch.from_numpy(entry).float()
            y = torch.from_numpy(y).float()
            
            if (self.use_gpu) is True:
                entry = entry.cuda()
                y = y.cuda()
                
            yhat = torch.matmul(entry,self.coefficients)
            
            # y - yhat for each element in tensor 
            difference = y - yhat
            
            # square differences
            difference_squared = torch.square(difference)
            
            if channelwise is True: 
                sum_difference = torch.sum(difference_squared, axis = 0)
            else: 
                sum_difference = torch.sum(difference_squared)
            
            # number of elements in matrix 
            rows, columns = y.shape
            if channelwise is True:
                n_elements = columns
            else: 
                n_elements = rows*columns
            
            # mean square error 
            mean_sq_error = sum_difference/n_elements
            
            # root mean square error
            rmse = torch.sqrt(mean_sq_error)
            
            return rmse.cpu().numpy()

        def concatenate_ones(self, X):
            # add an intercept to the multivariate regression
            ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
            return np.concatenate((ones, X), 1)
        
        def reshape_x(self, X):
            return X.reshape(-1, 1)
        
    # -------------------------------------------------------------------------
    # STEP 2.6 Define Correlation Function 
    # -------------------------------------------------------------------------

    def vectorized_correlation(x,y):
        dim = 0 # calculate the correlation for each channel 
        # we could additionally average the correlation over channels.
        
        # mean over all videos
        centered_x = x - x.mean(axis=dim, keepdims=True)
        centered_y = y - y.mean(axis=dim, keepdims=True)
    
        covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)
    
        bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
    
        # The addition of 1e-8 to x_std and y_std is commonly done 
        # to avoid division by zero or extremely small values.
        x_std = x.std(axis=dim, keepdims=True)+1e-8
        y_std = y.std(axis=dim, keepdims=True)+1e-8
    
        corr = bessel_corrected_covariance / (x_std * y_std)
    
        return corr.ravel()
    
    # -------------------------------------------------------------------------
    # STEP 2.7 Loop over all features and save best alpha hyperparameter
    # -------------------------------------------------------------------------
    alpha_tp = False # maybe add to function as a parameter above
    
    if input_type == 'miniclips':
        y_train, timepoints = load_eeg(sub, 'training', region, freq, input_type)       
    
    elif input_type == 'images':
        y_train, timepoints = load_eeg(sub, 'train', region, freq, input_type)        
        
    y_test, _ = load_eeg(sub, 'test', region, freq, input_type)
          
    output_names = ('rmse_score', 'correlation')
    
    # define matrix where to save the values 
    regression_features = dict.fromkeys(feature_names)
    
    for feature in features_dict.keys(): 
        print(feature)
        X_train, _ , X_test = load_features(feature, featuresDir)
        if alpha_tp is False: 
            alpha = load_alpha(sub, freq, region, feature, input_type)
        output = dict.fromkeys(output_names)
        
        rmse = np.zeros((timepoints, n_channels))
        corr = np.zeros((timepoints, n_channels))
        
        for tp in range(timepoints): 
            if alpha_tp is True: 
                alpha = load_alpha(sub, freq, region, feature, input_type)
            y_train_tp = y_train[:, :, tp]
            y_test_tp = y_test[:, :, tp]
            regression = OLS_pytorch(alpha = alpha)
            try: 
                regression.fit(X_train, y_train_tp, solver = 'cholesky')
            except Exception as error: 
                print("Attention. Cholesky solver did not work: ", error)
                print("Trying the standard linalg.solver...")
                regression.fit(X_train, y_train_tp, solver = 'solve')
            prediction = regression.predict(X_test)
            rmse_score = regression.score(entry = X_test, y = y_test_tp)
            correlation = vectorized_correlation(prediction, y_test_tp)
            rmse[tp, :] = rmse_score
            corr[tp, :] = correlation
        
        output['rmse_score'] = rmse
        output['correlation'] = corr
        regression_features[feature] = output
        
    # -------------------------------------------------------------------------
    # STEP 2.8 Save hyperparameters and scores
    # -------------------------------------------------------------------------    
    # Save the dictionary
    saveDir = f'/home/agnek95/Encoding-midlevel-features/Results/Encoding/{input_type}/7_features/'

    fileDir = (str(sub) + '_seq_' + str(freq) + 'hz_' + region +
               '_encoding_results_averaged_frame_before_mvnn_7features_onehot' + '.pkl')  
    
    savefileDir = os.path.join(saveDir, fileDir) 

    # Creating the directory if not existing
    if os.path.isdir(os.path.join(saveDir)) == False: # if not a directory
        os.makedirs(os.path.join(saveDir))
    
    with open(savefileDir, 'wb') as f:
        pickle.dump(regression_features, f)
    
    return regression_features

# -------------------------------------------------------------------------
# STEP 3 Run function
# ------------------------------------------------------------------------- 
if input_type == 'miniclips':
    subjects = [6, 7, 8, 9, 10, 11, 17, 18, 20, 21, 23, 25, 27, 28, 29, 30, 31, 32, 34, 36]
elif input_type == 'images':
    subjects = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

for sub in subjects:
    result = encoding(sub, freq, region, input_type) 
