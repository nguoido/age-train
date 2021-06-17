from pathlib import Path
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from src.factory import get_model, get_optimizer, get_scheduler, plot_result
from src.generator import ImageSequence

@hydra.main(config_path="src", config_name="config")
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks = [WandbCallback()]
    else:
        callbacks = []

    csv_path_test = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"{cfg.data.db_test}.csv")
    csv_path_train = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"{cfg.data.db_train}.csv")
    df_test = pd.read_csv(str(csv_path_test))
    df_train = pd.read_csv(str(csv_path_train))
    # train, val = train_test_split(df, random_state=42, test_size=0.1)
    train_gen = ImageSequence(cfg, df_train, "train")
    val_gen = ImageSequence(cfg, df_test, "val")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_model()
        print(">>>>>>>>>>>>>>>>>> START model.summary() >>>>>>>>>>>>>>>>>>")
        model.summary()
        print(">>>>>>>>>>>>>>>>>> STOP model.summary() >>>>>>>>>>>>>>>>>>")
        # opt = get_optimizer(cfg)
        scheduler = get_scheduler(cfg)
        # model.compile(optimizer=opt,
        #               loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
        #               metrics=['accuracy'])
        model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")
        print(">>>>>>>>>>>>>>>>>> START Model info >>>>>>>>>>>>>>>>>>")
        print(model.optimizer.get_config())
        print(">>>>>>>>>>>>>>>>>> STOP Model info >>>>>>>>>>>>>>>>>>")
        
    checkpoint_dir_save = "/content/drive/MyDrive/age_asian_3/checkpoint"
    filename = "_".join([cfg.model.model_name,
                         str(cfg.model.img_size),
                         "weights.{epoch:02d}-{val_loss:.2f}.hdf5"])
    callbacks.extend([
        LearningRateScheduler(schedule=scheduler),
        ModelCheckpoint(str(checkpoint_dir_save) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="max"),
        EarlyStopping(monitor="val_acc", mode="max", patience=10),
        ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)
    ])

    model.fit(train_gen, epochs=cfg.train.epochs, callbacks=callbacks, validation_data=val_gen,
              workers=multiprocessing.cpu_count())

    plot_result(model)

if __name__ == '__main__':
    main()
