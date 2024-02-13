from keras.applications import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D


class ResNet50FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self.create_feature_extractor()

    def create_feature_extractor(self):

        # include_top=False ile sınıflandırma katmanlarını çıkartıyoruz
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Removes last layer output from model
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

        return feature_extractor

    def get_model_summary(self):
        # Model summary
        self.model.summary()

class LightweightResNet50FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self.create_feature_extractor()

    def create_feature_extractor(self):

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Remove classification layers
        base_model.layers.pop()
        base_model.layers.pop()
        base_model.layers.pop()

        # Global Average Pooling Layer
        x = GlobalAveragePooling2D()(base_model.output)

        # new model
        feature_extractor = Model(inputs=base_model.input, outputs=x)

        return feature_extractor

    def get_model_summary(self):
        # Model summary
        self.model.summary()

class CustomResNet50FeatureExtractor:
    def __init__(self, selected_layers, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.selected_layers = selected_layers
        self.model = self.create_feature_extractor()

    def create_feature_extractor(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        selected_outputs = {}
        for layer_name in self.selected_layers:
            layer = base_model.get_layer(layer_name)
            selected_outputs[layer_name] = layer.output

        # New model with selected layers
        feature_extractor = Model(inputs=base_model.input, outputs=list(selected_outputs.values()))

        return feature_extractor

    def get_model_summary(self):
        self.model.summary()