import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, Compose, RandRotated, NormalizeIntensityd)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from monai.data.utils import pad_list_data_collate

dev_transforms = Compose(
    [
        LoadImaged(keys=["img"], image_only=True),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"]),
        RandRotated(keys=["img"], range_x=np.pi / 4, prob=0.2, keep_size=True, mode="nearest"),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["img"], image_only=False),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"]),
    ]
)

def map_sex_values(sex_array):
    """Utility function to map sex values if needed"""
    if "M" in sex_array or "F" in sex_array:
        return np.where(sex_array == "M", 1, 0)
    return sex_array

def load_dev_data(source_train_csv, source_val_csv,
              batch_size = 16, verbose = False):

    source_train = pd.read_csv(source_train_csv)
    source_val = pd.read_csv(source_val_csv)
    
    source_train_images = np.array(source_train["filename"]) #column containing path to each scan
    source_train_sex = map_sex_values(np.array(source_train["sex"])) #column containing sex groundtruths and map them to 0, 1
    source_train_id = np.array(source_train["id"]) #column containing subject id

    source_val_images = np.array(source_val["filename"])
    source_val_sex = map_sex_values(np.array(source_val["sex"]))
    source_val_id = np.array(source_val["id"])

    if verbose:
        print("Source train set size:", source_train_images.size, flush=True)
        print("Source train set sex size:", source_train_sex.size, flush=True)
        print("Source val set size:", source_val_images.size, flush=True)
        print("Source val set sex size:", source_val_sex.size, flush=True)
    
        assert np.all((source_train_sex == 0) | (source_train_sex == 1))
        assert np.all((source_val_sex == 0) | (source_val_sex == 1))
        
    # Putting the filenames in the MONAI expected format - source train set
    filenames_train_source = [{"img": x, "sex": int(y), "sid": z}\
                              for (x,y,z) in zip(source_train_images, source_train_sex, source_train_id)]
       
    source_ds_train = Dataset(filenames_train_source,
                                         dev_transforms)

    source_train_loader = DataLoader(source_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate)

    # Putting the filenames in the MONAI expected format - source val set
    filenames_val_source = [{"img": x, "sex": int(y), "sid": z}\
                              for (x,y,z) in zip(source_val_images, source_val_sex, source_val_id)]
       
    source_ds_val = Dataset(filenames_val_source,
                                         dev_transforms)
                                         
    source_val_loader = DataLoader(source_ds_val, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate)


    return source_ds_train, source_train_loader, source_ds_val, source_val_loader


def load_test_data(source_test_csv, verbose):
    source_test = pd.read_csv(source_test_csv)
    source_test_images = np.array(source_test["filename"])
    source_test_sex = map_sex_values(np.array(source_test["sex"]))
    source_test_id = np.array(source_test["id"])

    if verbose:
        print("Test Images Size: ", source_test_images.size, flush=True)
        print("Test Labels Size: ", source_test_sex.size, flush=True)

    # Putting the filenames in the MONAI expected format - source train set
    filenames_test_source = [{"img": x, "sex": y, "sid" : z}\
                                for (x,y,z) in zip(source_test_images, source_test_sex, source_test_id)]

    source_ds_test = Dataset(filenames_test_source, test_transforms)

    source_test_loader = DataLoader(source_ds_test,
                                    batch_size = 1,
                                    shuffle=False, 
                                    num_workers=0, 
                                    pin_memory=True)

    return source_ds_test, source_test_loader