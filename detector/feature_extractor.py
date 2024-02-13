import numpy as np
import os
import pandas as pd
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from matplotlib import gridspec

from resNet50 import ResNet50FeatureExtractor, LightweightResNet50FeatureExtractor, CustomResNet50FeatureExtractor

def custom_extract_features_from_csv(extractor, csv_path, base_image_path):
    df = pd.read_csv(csv_path)
    features_list = []
    for index, row in df.iterrows():
        image_path = os.path.join(base_image_path, row['image'])
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Feature extraction using the ResNet50 model
        features = feature_extraction(extractor, img_array)

        # Flatten the features and add them to the list
        flattened_features = [feature.flatten() for feature in features]
        features_list.append(flattened_features)

    # DataFrame with the extracted features
    features_df = pd.DataFrame(features_list)
    return features_df



def extract_features_from_csv(extractor, csv_path, base_image_path):
    df = pd.read_csv(csv_path)
    features_list = []
    for index, row in df.iterrows():
        image_path = os.path.join(base_image_path, row['image'])
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Feature extraction using the ResNet50 model
        features = feature_extraction(extractor, img_array)

        # Flatten the features and add them to the list
        flattened_features = features.flatten()
        features_list.append(flattened_features)

    # DataFrame with the extracted features
    features_df = pd.DataFrame(features_list)
    return features_df


def feature_extraction(extractor, img_array):
    features = extractor.model.predict(img_array)
    return features


def show_feature_map(features, img):
    n_features = features.shape[-1]

    fig = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    sub_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)

    for i in range(3):
        for j in range(3):
            ax2 = fig.add_subplot(sub_gs[i, j])
            plt.axis('off')
            plt.imshow(features[0, :, :, np.random.randint(n_features)], cmap='viridis')

    plt.show()


def main():

    train_csv_path = "dataset/train_similar.csv"
    test_csv_path = "dataset/test_similar.csv"
    base_image_path = "../deepFakeArtChallenge/v2/"

    # For CustomResNet50FeatureExtractor
    selected_layers = ['conv1_conv', 'conv2_block1_out', 'conv3_block1_out']
    resnet_extractor = CustomResNet50FeatureExtractor(selected_layers=selected_layers)

    train_features = custom_extract_features_from_csv(resnet_extractor, train_csv_path, base_image_path)
    train_features.to_csv("train_features_custom_layers.csv", index=False)

    test_features = custom_extract_features_from_csv(resnet_extractor, test_csv_path, base_image_path)
    test_features.to_csv("test_features_custom_layers.csv", index=False)

    print("Eğitim veri kümesi özellik vektörlerinin boyutu:", train_features.shape)
    print("Test veri kümesi özellik vektörlerinin boyutu:", test_features.shape)


if __name__ == "__main__":
    main()




