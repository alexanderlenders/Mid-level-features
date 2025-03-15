#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENCODING - DEEP NETS

This script implements the multivariate linear ridge regression for the EEG
data. 

@author: Alexander Lenders, Agnessa Karapetian
"""
import os
import numpy as np
import torch
import pickle
import argparse

def load_activation(input_type, img_type, layer_id):
    if input_type == 'images':
        layer_dir = "/scratch/agnek95/Unreal/CNN_activations_redone/2D_ResNet18/pca_90_percent/prepared/"
    elif input_type == 'miniclips':
        layer_dir = "/scratch/agnek95/Unreal/CNN_activations_redone/3D_ResNet18/pca_90_percent/prepared/"

    fileDir = f"{layer_id}_layer_activations_" + img_type + ".npy"
    total_dir = os.path.join(layer_dir, fileDir)

    # Load EEG data
    y = np.load(total_dir, allow_pickle=True)

    return y

def load_features(feature, featuresDir):
    """
    Load low/mid/high-level feature annotations.
    """

    features = np.load(featuresDir, allow_pickle=True)
    X_prep = features[feature]

    X_train = X_prep[0]
    X_val = X_prep[1]
    X_test = X_prep[2]

    return X_train, X_val, X_test

class OLS_pytorch(object):
    """
    Class for solving the ridge regression with OLS.
    """

    def __init__(
        self, use_gpu: bool = False, intercept: bool = True, alpha: float = 0
    ):
        self.coefficients = []
        self.use_gpu = use_gpu
        self.intercept = intercept
        self.alpha = alpha  # ridge penalty

    def fit(self, X: np.array, y: np.array, solver: str = "cholesky"):
        """
        Fit function for solving the ridge regression with OLS.
        Details (Statistical approach):
            https://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/LinearRegression.pdf
        For skeleton position, we have to use the cholesky solver since the
        Hermetian matrix is not positive definit for the skeleton position.
        There are different solvers for ridge regression, each of them
        have their advantages.
        - Choleksy decomposition
        - LU decomposition
        - Fore more details, refer to:
            https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
        """
        if len(X.shape) == 1:
            X = self.reshape_x(X)
        if len(y.shape) == 1:
            y = self.reshape_x(y)

        # Add intercept by adding a column of ones
        if (self.intercept) is True:
            X = self.concatenate_ones(X)

        # convert numpy array into torch
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        # if we use a gpu, we have to transfer the torch to it
        if (self.use_gpu) is True:
            X = X.cuda()
            y = y.cuda()

        _, columns = X.shape
        _, columns_y = y.shape

        # Use data augmentation trick to solve the ridge regression
        # As described in Hastie (2020)
        penalty_matrix = np.eye((columns))

        # Since we don't want to penalize the intercept
        penalty_matrix[0, 0] = 0

        penalty_matrix = torch.from_numpy(
            penalty_matrix * np.sqrt(self.alpha)
        ).float()

        zero_matrix = torch.from_numpy(np.zeros((columns, columns_y))).float()

        if (self.use_gpu) is True:
            penalty_matrix = penalty_matrix.cuda()
            zero_matrix = zero_matrix.cuda()

        X = torch.vstack((X, penalty_matrix))
        y = torch.vstack((y, zero_matrix))

        # Creates Hermitian positive-definite matrix
        XtX = torch.matmul(X.t(), X)
        Xty = torch.matmul(X.t(), y)

        # Solve it
        if solver == "cholesky":
            # Cholesky decomposition, creates the lower triangle matrix
            L = torch.linalg.cholesky(XtX)
            self.coefficients = torch.cholesky_solve(Xty, L)

        elif solver == "lstsq":
            lstsq_coefficients, _, _, _ = torch.linalg.lstsq(
                Xty, XtX, rcond=None
            )
            self.coefficients = lstsq_coefficients.t()

        elif solver == "solve":
            # Assumes that the matrix is invertible
            self.coefficients = torch.linalg.solve(XtX, Xty)

    def predict(self, entry):
        # entry refers to the features of the test data
        entry = self.concatenate_ones(entry)
        entry = torch.from_numpy(entry).float()

        if (self.use_gpu) is True:
            entry = entry.cuda()

        prediction = torch.matmul(entry, self.coefficients)
        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction)

        return prediction

    def score(self, entry, y, channelwise=True):
        """
        Computes RMSE for ridge regression.
        """

        entry = self.concatenate_ones(entry)

        entry = torch.from_numpy(entry).float()
        y = torch.from_numpy(y).float()

        if (self.use_gpu) is True:
            entry = entry.cuda()
            y = y.cuda()

        yhat = torch.matmul(entry, self.coefficients)

        # y - yhat for each element in tensor
        difference = y - yhat

        # square differences
        difference_squared = torch.square(difference)

        if channelwise is True:
            sum_difference = torch.sum(difference_squared, axis=0)
        else:
            sum_difference = torch.sum(difference_squared)

        # number of elements in matrix
        rows, columns = y.shape
        if channelwise is True:
            n_elements = columns
        else:
            n_elements = rows * columns

        # mean square error
        mean_sq_error = sum_difference / n_elements

        # root mean square error
        rmse = torch.sqrt(mean_sq_error)

        return rmse.cpu().numpy()

    def concatenate_ones(self, X):
        # add an intercept to the multivariate regression in first column
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)

    def reshape_x(self, X):
        return X.reshape(-1, 1)

def load_alpha(input_type,feature):
    """
    Load optimal alpha hyperparameter for ridge regression.
    """
    if input_type == 'images':
        file_part = '2D'
    elif input_type == 'miniclips':
        file_part = '3D'
    alphaDir = f'/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/{file_part}_ResNet18/pca_90_percent/hyperparameters/'

    alpha_values = np.load(alphaDir, allow_pickle=True)

    alpha = alpha_values[feature]["best_alpha_a_corr"]

    return alpha

def vectorized_correlation(x, y):
    dim = 0  # calculate the correlation for each channel

    # mean over all videos
    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    # The addition of 1e-8 to x_std and y_std is commonly done
    # to avoid division by zero or extremely small values.
    x_std = x.std(axis=dim, keepdims=True) + 1e-8
    y_std = y.std(axis=dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()


def encoding(input_type):
    """
    Perform encoding (ridge regression) for predicting the unit activations
    in deep nets.

    Parameters
    ----------
    input_type : str
        Images or miniclips

    """
    # -------------------------------------------------------------------------
    # STEP 1 Define Variables
    # -------------------------------------------------------------------------
    layers_names = (
        "layer1.0.relu_1",
        "layer1.1.relu_1",
        "layer2.0.relu_1",
        "layer2.1.relu_1",
        "layer3.0.relu_1",
        "layer3.1.relu_1",
        "layer4.0.relu_1",
        "layer4.1.relu_1",
    )
    feature_names = (
        "edges",
        "world_normal",
        "lighting",
        "scene_depth",
        "reflectance",
        "action",
        "skeleton",
    )


    if input_type == 'images':
        featuresDir = '/home/agnek95/Encoding-midlevel-features/Results/Encoding/images/7_features/img_features_frame_20_redone_7features_onehot.pkl'
        explained_var_dir = f'/scratch/agnek95/Unreal/CNN_activations_redone/2D_ResNet18/pca_90_percent/pca/'
        saveDir = '/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/2D_ResNet18/pca_90_percent/encoding/'
        alpha_dir = '/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/2D_ResNet18/pca_90_percent/hyperparameters/'

    elif input_type == 'miniclips':
        featuresDir = '/home/agnek95/Encoding-midlevel-features/Results/Encoding/miniclips/7_features/video_features_avg_frame_redone.pkl'
        explained_var_dir = f'/scratch/agnek95/Unreal/CNN_activations_redone/3D_ResNet18/pca_90_percent/pca/'
        saveDir = '/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/3D_ResNet18/pca_90_percent/encoding/'
        alpha_dir = '/home/agnek95/Encoding-midlevel-features/Results/CNN_Encoding/3D_ResNet18/pca_90_percent/hyperparameters/'

    features_dict = dict.fromkeys(feature_names)

    # Device agnostic code: Use gpu if possible, otherwise cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # STEP 2 Loop over all features and save best alpha hyperparameter
    # -------------------------------------------------------------------------
    output_names = (
        "rmse_score",
        "correlation",
        "rmse_average",
        "correlation_average",
    )

    # define matrix where to save the values
    regression_features = dict.fromkeys(feature_names)

    num_layers = len(layers_names)

    for feature in features_dict.keys():
        print(feature)
        X_train, _, X_test = load_features(feature, featuresDir)

        if explained_var_dir:
            alpha_dir_final = os.path.join(alpha_dir, "weighted")
        else:
            alpha_dir_final = os.path.join(alpha_dir, "unweighted")

        alpha = load_alpha(
            alphaDir=alpha_dir_final, feature=feature, eeg=False
        )

        output = dict.fromkeys(output_names)

        rmse_scores = {}
        corr_scores = {}

        for tp, l in enumerate(layers_names):
            print(l)

            y_train_tp = load_activation(input_type,"training", l)
            y_test_tp = load_activation(input_type, "test", l)

            regression = OLS_pytorch(alpha=alpha)
            try:
                regression.fit(X_train, y_train_tp, solver="cholesky")
            except Exception as error:
                print("Attention. Cholesky solver did not work: ", error)
                print("Trying the standard linalg.solver...")
                regression.fit(X_train, y_train_tp, solver="solve")
            prediction = regression.predict(X_test)
            rmse_score = regression.score(entry=X_test, y=y_test_tp)
            correlation = vectorized_correlation(prediction, y_test_tp)

            rmse_scores[l] = rmse_score
            corr_scores[l] = correlation

        if explained_var_dir:
            rmse_avg_chan = np.zeros((num_layers))
            corr_avg_chan = np.zeros((num_layers))

            for i, layer in enumerate(layers_names):

                explained_var_dir_layer = os.path.join(
                    explained_var_dir, layer, "explained_variance.pkl"
                )

                with open(explained_var_dir_layer, "rb") as file:
                    explained_var = pickle.load(file)

                explained_var = np.array(explained_var["explained_variance"])
                total_variance = np.sum(explained_var)

                rmse_it = rmse_scores[layer]
                corr_it = corr_scores[layer]

                rmse_avg_chan[i] = (
                    np.sum(rmse_it * explained_var) / total_variance
                )
                corr_avg_chan[i] = (
                    np.sum(corr_it * explained_var) / total_variance
                )

        output["rmse_score"] = rmse_scores
        output["correlation"] = corr_scores
        output["rmse_average"] = rmse_avg_chan
        output["correlation_average"] = corr_avg_chan

        regression_features[feature] = output

    # -------------------------------------------------------------------------
    # STEP 3 Save results
    # -------------------------------------------------------------------------
    # Save the dictionary
    fileDir = "encoding_layers_resnet.pkl"

    if explained_var_dir:
        resultsDir = os.path.join(saveDir, "weighted")
    else:
        resultsDir = os.path.join(saveDir, "unweighted")

    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    savefileDir = os.path.join(resultsDir, fileDir)

    with open(savefileDir, "wb") as f:
        pickle.dump(regression_features, f)

    return regression_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_type",
        default='images',
        type=str,
        help='Images or miniclips',
        required=True
    )
  
    args = parser.parse_args()

    input_type = args.input_type
    encoding(input_type)
