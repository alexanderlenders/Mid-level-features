"""
Utility functions for the EEG encoding analysis.
"""

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import cv2 as cv
from scipy.ndimage import convolve, gaussian_filter
from PIL import Image
import numpy as np
import torch
import os
import ast
from configparser import ConfigParser
import pandas as pd


def canny_edge(
    image: int,
    images_dir: str,
    frame: int,
    openCV: bool = True,
    gaussian_filter_size: int = 3,
):
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

    Details
    ---------
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
    # Import image
    if "images" in images_dir:
        image_file = (
            str(image).zfill(4) + "_frame_{}".format(frame) + ".jpg"
        )  # zfill: fill with zeros (4)
    else:
        image_file = (
            str(image).zfill(4) + "_default_frame_{}".format(frame) + ".jpg"
        )  # zfill: fill with zeros (4)

    image_dir = images_dir + "/" + image_file

    if openCV is True:
        img = cv.imread(image_dir)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # It is recommended to blur before doing canny edge detection
        blur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

        # Recommended thresholds in a stack overflow question
        # high_threshold = 255
        # low_threshold = high_threshold/3

        # Values as in YouTube Tutorial:
        canny_edges = cv.Canny(blur, 125, 175)

    elif openCV is False:
        img = Image.open(image_dir)
        img = np.array(img.convert("L")).astype(np.float32)
        # L means we convert it to a grey-valued image

        # Gaussian blur to reduce noise level
        gaussian_image = gaussian_filter(img, gaussian_filter_size)
        # We could also try out a 5x5x5 filter (as recommended in the openCV
        # tutorial)

        # Use sobel filters to get gradients with respect to x and y
        grad_x = convolve(gaussian_image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        grad_y = convolve(gaussian_image, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Compute the magnitude of the gradient (:= edge strength)
        canny_edges = np.power(
            np.power(grad_x, 2.0) + np.power(grad_y, 2.0), 0.5
        )

        # Compute the edge direction
        # theta = np.arctan2(grad_y, grad_x)

        # What is missing here in comparison to the openCV implementation is
        # the non-maximum surpression and hysteresis thresholding

    return canny_edges


def action(image: int, action_data: pd.DataFrame, actions: list):
    """
    Parameters
    ----------
    image: int
        Number of image
    """
    action_id = action_data.iloc[image, 1]
    one_hot_vector = np.zeros(
        [
            len(actions),
        ]
    )
    one_hot_vector[action_id] = 1
    return one_hot_vector


def skeleton_pos(image: int, annotations_dir: str, frame: int):
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
    image_file = (
        str(image).zfill(4)
        + "_skeleton_position_frame_{}".format(frame)
        + ".pkl"
    )  # zfill: fill with zeros (4)
    image_dir = annotations_dir + "/" + image_file
    pickle = np.load(image_dir, allow_pickle=True)
    # Extract bone names
    # bone_names = pickle['bone_name'].tolist()
    position_x_y = np.array(pickle[["screen_pos_x", "screen_pos_y"]])
    return position_x_y


def world_normals(image: int, annotations_dir: str, frame: int):
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
    image_file = (
        str(image).zfill(4)
        + "_world_normal"
        + "_frame_{}".format(frame)
        + ".jpg"
    )
    image_dir = annotations_dir + "/" + image_file
    image = Image.open(image_dir)
    world_normals_rgb = np.array(image.convert("RGB")).astype(np.float32)
    return world_normals_rgb


def lighting(image: int, annotations_dir: str, frame: int):
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
    if "images" in annotations_dir:
        image_file = (
            str(image).zfill(4)
            + "_lightning"
            + "_frame_{}".format(frame)
            + ".jpg"
        )
    else:
        image_file = (
            str(image).zfill(4)
            + "_lighting"
            + "_frame_{}".format(frame)
            + ".jpg"
        )
    image_dir = annotations_dir + "/" + image_file
    image = Image.open(image_dir)
    lighting_np = np.array(image.convert("L")).astype(np.float32)
    return lighting_np


def depth(image: int, annotations_dir: str, frame: int):
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
    image_file = (
        str(image).zfill(4)
        + "_scene_depth"
        + "_frame_{}".format(frame)
        + ".jpg"
    )
    image_dir = annotations_dir + "/" + image_file
    image = Image.open(image_dir)
    scene_depth = np.array(image.convert("L")).astype(np.float32)
    return scene_depth


def reflectance(image: int, annotations_dir: str, frame: int):
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
    image_file = (
        str(image).zfill(4)
        + "_reflectance"
        + "_frame_{}".format(frame)
        + ".jpg"
    )
    image_dir = annotations_dir + "/" + image_file
    image = Image.open(image_dir)
    reflectance_rgb = np.array(image.convert("RGB")).astype(np.float32)
    return reflectance_rgb


def pca(
    features_train: np.ndarray,
    features_val: np.ndarray,
    features_test: np.ndarray,
    pca_method: str = "linear",
    n_comp: int = 100,
):
    """
    Note: This implements a (simple) PCA using SVD. One could also implement
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
    if pca_method == "linear":
        pca_image = PCA(n_components=n_comp, random_state=42)
    elif pca_method == "nonlinear":
        pca_image = KernelPCA(
            n_components=n_comp, kernel="poly", degree=4, random_state=42
        )

    pca_image.fit(scaled_train)

    # if pca_method == 'linear':
    # Get explained variance
    # per_var = np.round(pca_image.explained_variance_ratio_* 100, decimals=1)
    # labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    # explained_variance = np.sum(per_var)

    # Transform data
    pca_train = pca_image.transform(scaled_train)
    pca_test = pca_image.transform(scaled_test)
    pca_val = pca_image.transform(scaled_val)

    return pca_train, pca_val, pca_test


def load_eeg(
    sub: int,
    img_type: str,
    region: str,
    freq: int,
    input_type: str,
    eeg_dir: str,
):
    """
    Utility function to load the EEG data for a given subject and input type (video or image).
    """

    # Define the directory
    workDirFull = eeg_dir

    # load mvnn data
    if input_type == "miniclips":
        if sub < 10:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/sub-0{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + img_type
                + "/"
                + region
                + "/",
            )
            fileDir = (
                "sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                    sub, img_type, freq, region
                )
            )

        else:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/sub-{}".format(sub)
                + "/eeg/preprocessing/ica"
                + "/"
                + img_type
                + "/"
                + region
                + "/",
            )
            fileDir = (
                "sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                    sub, img_type, freq, region
                )
            )

    elif input_type == "images":
        if sub < 10:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-0{}".format(sub)
                + "/{}/{}/{}hz/".format(img_type, region, freq),
            )
            fileDir = (
                "sub-0{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                    sub, img_type, freq, region
                )
            )
        else:
            folderDir = os.path.join(
                workDirFull,
                "{}".format(input_type)
                + "/prepared"
                + "/sub-{}".format(sub)
                + "/{}/{}/{}hz/".format(img_type, region, freq),
            )
            fileDir = (
                "sub-{}_seq_{}_{}hz_{}_prepared_epochs_redone.npy".format(
                    sub, img_type, freq, region
                )
            )

    total_dir = os.path.join(folderDir, fileDir)

    # Load EEG data
    data = np.load(total_dir, allow_pickle=True).item()

    eeg_data = data["eeg_data"]
    img_cat = data["img_cat"]

    del data

    # Average over trials
    if input_type == "miniclips":
        n_conditions = len(np.unique(img_cat))
        _, n_channels, timepoints = eeg_data.shape
        n_trials = img_cat.shape[0]
        n_rep = round(n_trials / n_conditions)

        y_prep = np.zeros(
            (n_conditions, n_rep, n_channels, timepoints), dtype=float
        )

        for condition in range(n_conditions):
            idx = np.where(img_cat == np.unique(img_cat)[condition])
            y_prep[condition, :, :, :] = eeg_data[idx, :, :]
    elif input_type == "images":
        _, n_channels, timepoints = eeg_data.shape
        if img_type == "train":
            n_conditions = 1080
            n_rep = 5
        elif img_type == "test":
            n_conditions = 180
            n_rep = 30
        elif img_type == "val":
            n_conditions = 180
            n_rep = 5
        y_prep = eeg_data.reshape(n_conditions, n_rep, n_channels, timepoints)

    y = np.mean(y_prep, axis=1)

    return y, timepoints


def load_features(feature: str, featuresDir: str):
    """
    Utility function to load the preprocessed features for a given feature type.
    """
    features = np.load(featuresDir, allow_pickle=True)
    X_prep = features[feature]

    X_train = X_prep[0]
    X_val = X_prep[1]
    X_test = X_prep[2]

    return X_train, X_val, X_test


def load_feature_set(feature_set: str, featuresDir: str):
    """
    Loads and concatenates multiple features.
    """
    if isinstance(feature_set, str):
        X_train, X_val, X_test = load_features(feature_set, featuresDir)
        return X_train, X_val, X_test
    else:
        features_dict = np.load(featuresDir, allow_pickle=True)

        X_train_list = []
        X_val_list = []
        X_test_list = []

        for fname in feature_set:
            X_prep = features_dict[fname]  # tuple: (train, val, test)
            X_train_list.append(X_prep[0])
            X_val_list.append(X_prep[1])
            X_test_list.append(X_prep[2])

        # Concatenate along feature dimension (axis=1)
        X_train = np.concatenate(X_train_list, axis=1)
        X_val = np.concatenate(X_val_list, axis=1)
        X_test = np.concatenate(X_test_list, axis=1)

        return X_train, X_val, X_test


def load_alpha(
    sub: int,
    freq: int,
    region: str,
    feature: str,
    input_type: str,
    feat_dir: str,
    tp: int = None,
    feat_len: int = 7,
):
    """
    Utility function to load the optimized alpha value for a given subject.
    """
    savedDir = os.path.join(feat_dir, input_type)

    fileDir = (
        str(sub)
        + "_seq_"
        + str(freq)
        + "hz_"
        + region
        + f"_hyperparameter_tuning_averaged_frame_before_mvnn_{feat_len}_features_onehot"
        + ".pkl"
    )

    alphaDir = os.path.join(savedDir, fileDir)

    alpha_values = np.load(alphaDir, allow_pickle=True)

    if tp:
        if isinstance(feature, list):
            alpha = alpha_values[(", ".join(feature))]["best_alpha_corr"]
            alpha = alpha[tp]
        else:
            alpha = alpha_values[feature]["best_alpha_corr"]
            alpha = alpha[tp]
    else:
        if isinstance(feature, list):
            alpha = alpha_values[(", ".join(feature))]["best_alpha_a_corr"]
        else:
            alpha = alpha_values[feature]["best_alpha_a_corr"]

    return alpha


class OLS_pytorch(object):
    def __init__(self, use_gpu=False, intercept=True, ridge=True, alpha=0):
        self.coefficients = []
        self.use_gpu = use_gpu
        self.intercept = intercept
        self.ridge = ridge
        self.alpha = alpha  # penalty (alpha or lambda)

    def fit(self, X, y, solver="cholesky"):
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
            _, columns_y = y.shape

            # we use the data augmentation approach to solve the ridge
            # regression via OLS

            penalty_matrix = np.eye((columns))
            penalty_matrix = torch.from_numpy(
                penalty_matrix * np.sqrt(self.alpha)
            ).float()

            zero_matrix = torch.from_numpy(
                np.zeros((columns, columns_y))
            ).float()

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
                # Choleksy decomposition, creates the lower triangle matrix
                L = torch.linalg.cholesky(XtX)

                betas_cholesky = torch.cholesky_solve(Xty, L)

                self.coefficients = betas_cholesky
                return betas_cholesky

            elif solver == "lstsq":
                lstsq_coefficients, _, _, _ = torch.linalg.lstsq(
                    Xty, XtX, rcond=None
                )
                return lstsq_coefficients.t()
                self.coefficients = lstsq_coefficients.t()

            elif solver == "solve":
                solve_coefficients = torch.linalg.solve(XtX, Xty)
                self.coefficients = solve_coefficients

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
        # add an intercept to the multivariate regression
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)

    def reshape_x(self, X):
        return X.reshape(-1, 1)


def vectorized_correlation(x: np.ndarray, y: np.ndarray):
    dim = 0  # calculate the correlation for each channel
    # we could additionally average the correlation over channels.

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


def parse_list(value):
    """
    Parse a list-like string into a list.
    """
    try:
        return ast.literal_eval(value)
    except ValueError:
        print("Could not parse list-like string.")
        raise ValueError


def load_config(config_dir: str, section: str):
    """
    Utility function to load a configuration file.
    """
    config = ConfigParser()
    config.read(config_dir)

    if not config.has_section(section):
        print(f"Current directory: {os.getcwd()}")
        print(f"Reading configuration file: {config_dir}")
        raise ValueError(
            f"Section {section} not found {config_dir}, check the config file and directory."
        )

    return config
