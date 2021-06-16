[1mdiff --git a/re_age_asian_train.py b/re_age_asian_train.py[m
[1mindex 54127ee..3530656 100644[m
[1m--- a/re_age_asian_train.py[m
[1m+++ b/re_age_asian_train.py[m
[36m@@ -35,7 +35,8 @@[m [mdef main(cfg):[m
 [m
     with strategy.scope():[m
         # model = get_model(cfg)[m
[31m-        model = load_model("/content/drive/MyDrive/age_asian_2/checkpoint2/EfficientNetB3_224_weights.04-0.55.hdf5")[m
[32m+[m[32m        model = load_model("/content/drive/MyDrive/age_asian_2/checkpoint3/EfficientNetB3_224_weights.07-0.76.hdf5[m
[32m+[m[32mEpoch 8/30")[m
         # opt = get_optimizer(cfg)[m
         value_lr = K.get_value(model.optimizer.learning_rate)[m
         print(">>>>>>>>>>>>>>>>>> learning rate: {} >>>>>>>>>>>>>>>>>>".format(value_lr))[m
[36m@@ -49,7 +50,7 @@[m [mdef main(cfg):[m
         print(model.optimizer.get_config())[m
         print(">>>>>>>>>>>>>>>>>> STOP Model info >>>>>>>>>>>>>>>>>>")[m
 [m
[31m-    checkpoint_dir_save = "/content/drive/MyDrive/age_asian_2/checkpoint3"[m
[32m+[m[32m    checkpoint_dir_save = "/content/drive/MyDrive/age_asian_2/checkpoint4"[m
     filename = "_".join([cfg.model.model_name,[m
                          str(cfg.model.img_size),[m
                          "weights.{epoch:02d}-{val_loss:.2f}.hdf5"])[m
