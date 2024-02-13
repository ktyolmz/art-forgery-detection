# Art Forgery Detection using Machine Learning

This repository contains a machine learning project for detecting AI-generated art images. It utilizes ResNet50 for feature extraction and logistic regression for classification. [DeepFakeArt Challenge](https://github.com/h-aboutalebi/DeepfakeArt)
dataset is used as the dataset for training and evaluation. More details can be found on research paper named "A Machine Learning Approach to Unmasking Art Forgery Detection and Analysis".

## Overview

- **DeepFakeArt Challenge Dataset**: The dataset comprises genuine and forged artwork images, enabling robust training and evaluation of the model.
- **Feature Extraction with ResNet50**: ResNet50 architecture extracts features from genuine and fake artwork images.
- **Logistic Regression Classifier**: Features extracted from ResNet50 are fed into a logistic regression classifier for determining whether an artwork is authentic or forged.

## Project Structure

The repository is structured into two main packages:

1. **DataPreprocessing**: Contains scripts for preprocessing the dataset. Tasks include data cleaning, modifying train and test set files, and organizing the dataset.
2. **Detector**: It contains ResNet50 models for feature extraction, a Logistic Regression Model for model training, and model evaluation scripts.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- DeepFakeArt Challenge for providing the dataset.
- TenserFlow, pandas and scikit-learn libraries for their implementations used in this project.
