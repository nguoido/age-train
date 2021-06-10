from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from keras.applications.resnet50 import ResNet50

# def get_model(cfg):
#     base_model = None
#     base_model = ResNet50(
#                     include_top=False,
#                     weights='imagenet',
#                     input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
#                     pooling="avg")
#     prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
#                        name="pred_age")(base_model.output)
#     model = Model(inputs=base_model.input, outputs=prediction)
#     return model


# megaage_asian
def get_model(cfg):
    base_model = None
    base_model = ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
                    pooling="avg")
    features = base_model.output
    pred_age = Dense(units=70, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=pred_age)

    return model



def get_optimizer(cfg):
    if cfg.train.optimizer_name == "sgd":
        return SGD(learning_rate=cfg.train.lr, momentum=0.9, nesterov=True)
    elif cfg.train.optimizer_name == "adam":
        return Adam(learning_rate=cfg.train.lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def get_scheduler(cfg):
    class Schedule:
        def __init__(self, nb_epochs, initial_lr):
            self.epochs = nb_epochs
            self.initial_lr = initial_lr

        def __call__(self, epoch_idx):
            if epoch_idx < self.epochs * 0.25:
                return self.initial_lr
            elif epoch_idx < self.epochs * 0.50:
                return self.initial_lr * 0.2
            elif epoch_idx < self.epochs * 0.75:
                return self.initial_lr * 0.04
            return self.initial_lr * 0.008
    return Schedule(cfg.train.epochs, cfg.train.lr)
