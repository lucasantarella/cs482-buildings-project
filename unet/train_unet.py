from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
import uuid
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 3
N_CLASSES = 1  # buildings
CLASS_WEIGHTS = [1]
N_EPOCHS = 2500
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 50
TRAIN_SZ = 800  # train size
VAL_SZ = 200    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = "/home/jovyan/work/unet/weights"
if not os.path.exists(weights_path):
    os.mkdir(weights_path)
    print("Made Weights Folder")
else:
    print("Weights Folder Already Exists")
model_uuid = str(uuid.uuid4())
weights_path += '/' + model_uuid + 'unet_weights.hdf5'
trainIds = os.listdir('/data/train/RGB-PanSharpen/')[:500]


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()
    print("getting model")
    
    print('Reading images')
    count = 0
    for img_id in trainIds:
        #img_m = normalize(tiff.imread('/data/train/RGB-PanSharpen/{}'.format(img_id)).transpose([1, 2, 0]))
        #mask = tiff.imread('/data/train/Masks/buildings_{}'.format(img_id[15:])).transpose([1, 2, 0]) / 255
        img_m = normalize(tiff.imread('/data/train/RGB-PanSharpen/{}'.format(img_id)))
        mask = np.array([tiff.imread('/data/train/Masks/buildings_{}'.format(img_id[15:]))/ 255]).transpose([1, 2, 0])
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        count+=1
        print(str(count)+"\t"+img_id + ' read')
        
    print('Images were read')
    
    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    print('Training model', model_uuid)
    train_net()
    print(model_uuid, "Model Saved. Please Check your directory")
