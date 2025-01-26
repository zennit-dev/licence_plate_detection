import keras
from keras import Input
from keras.src.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.settings import config


class SetUpModel:
    """Set up the model"""

    @staticmethod
    def __build_conv_blocks(
        inputs: keras.layers.Layer, include_augmentation: bool = True
    ) -> keras.layers.Layer:
        """Build convolutional blocks dynamically from config."""
        # Create preprocessing layers list - separate augmentation from basic preprocessing
        basic_preprocessing = [keras.layers.Rescaling(1.0 / 255)]
        augmentation_layers = [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
        ]

        # Create preprocessing model with or without augmentation
        preprocessing_layers = basic_preprocessing + (
            augmentation_layers if include_augmentation else []
        )
        preprocessing_model = keras.Sequential(preprocessing_layers, name="preprocessing")
        x = preprocessing_model(inputs)

        # Build conv blocks layers list
        conv_layers = []
        if hasattr(config.model.layer, "conv_blocks"):
            for _, block_config in config.model.layer.conv_blocks.items():
                for filters in block_config.filters:
                    conv_layers.append(
                        Conv2D(
                            filters=filters,
                            kernel_size=block_config.kernel_size,
                            activation=config.model.hyperparameters.activation,
                            padding="same",
                        )
                    )
                conv_layers.append(MaxPooling2D(pool_size=block_config.pool_size))
                conv_layers.append(Dropout(config.model.hyperparameters.dropout_rate))

        # Create conv blocks model if we have layers, otherwise just return input
        if conv_layers:
            conv_model = keras.Sequential(conv_layers, name="conv_blocks")
            return conv_model(x)
        return x

    @staticmethod
    def __build_dense_layers(inputs: keras.layers.Layer) -> keras.layers.Layer:
        """Build dense layers dynamically from config."""
        x = Flatten()(inputs)

        for layer_name, layer_config in config.model.layer.dense_layers.items():
            x = Dense(units=layer_config.units, activation=config.model.hyperparameters.activation)(
                x
            )
            x = Dropout(layer_config.dropout_rate)(x)

        return x

    @staticmethod
    def __build_optimizer() -> keras.optimizers.Optimizer:
        match config.model.architecture.optimizer.lower():
            case "adam":
                return keras.optimizers.Adam(
                    learning_rate=config.optimizer.adam.learning_rate,
                    beta_1=config.optimizer.adam.beta_1,
                    beta_2=config.optimizer.adam.beta_2,
                    epsilon=config.optimizer.adam.epsilon,
                    amsgrad=config.optimizer.adam.amsgrad,
                )
            case "sgd":
                return keras.optimizers.SGD(
                    learning_rate=config.optimizer.sgd.learning_rate,
                    momentum=config.optimizer.sgd.momentum,
                    nesterov=config.optimizer.sgd.nesterov,
                )
            case "rmsprop":
                return keras.optimizers.RMSprop(
                    learning_rate=config.optimizer.rmsprop.learning_rate,
                    rho=config.optimizer.rmsprop.rho,
                    momentum=config.optimizer.rmsprop.momentum,
                    epsilon=config.optimizer.rmsprop.epsilon,
                    centered=config.optimizer.rmsprop.centered,
                )

    def create_model(self, num_classes: int, include_augmentation: bool = True) -> keras.Model:
        """Create and compile the model based on configuration."""
        inputs = Input(shape=config.model.hyperparameters.input_shape)
        x = self.__build_conv_blocks(inputs, include_augmentation)
        x = self.__build_dense_layers(x)
        outputs = Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.__build_optimizer(),
            loss=config.model.architecture.loss,
            metrics=config.model.architecture.metrics,
        )
        return model
