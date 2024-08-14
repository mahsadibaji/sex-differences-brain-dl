# Sex Classification
### Saliency Map
![](https://github.com/mahsadibaji/sex-differences-brain-dl/blob/main/tiv-saliency-map.gif)
### Source Code Structure
- ***data*** folder contains csv sheets for train, validation and test sets. Each file includes the dataset name, filename, subject id, age, sex, vendor, and magnetic field strength related to each brain scan. Please replace the generic values in `filename` column with correct path to each file.
- ***scripts*** folder contains code for loading data, training, and testing. It is recommended to go through the code and fill out place holders according to your needs.
- ***pretrained-weights*** folder contains the optimized trained model used in the paper for reporting the results.
- ***saliency-maps*** folder contains the niftii files of saliency maps showing sex-specific regions identified by DL model. The maps are averaged from maps related to correctly classified samples in the test set.
- `requirements.yml` contains package dependencies.
(data-section)=
### Downloading the Data
- CC359 - [data access](https://www.ccdataset.com/download)
- CamCAN - [data access](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
- ADNI - [data access](https://adni.loni.usc.edu/data-samples/access-data/)
- OASIS-3 - [data access](https://www.oasis-brains.org/#data)
### Training the Model
To start training the model, navigate to your project directory and run the run_train.py script with the required parameters (make sure to specify the directory to save results). Below is the command:
```
python /scripts/run_train.py --batch_size 16 --learning_rate 0.01 --epochs 50 --results_dir <path_to_results_dir> --source_train_csv /data/train_data.csv --source_val_csv /data/val_data.csv --verbose <True/False>
```

### Testing the model
To test the model, run the run_test.py script with the required parameters (make sure to specify the directory to save results). Below is the command:
```
python /scripts/run_test.py --results_dir <path_to_results_dir> --source_test_csv /data/test_data.csv --saved_model_path /pretrained-weights/sfcn_best.pth --verbose <True/False>
```

### Arguments
- batch_size (int): The size of the batch used in training.
- learning_rate (float): The learning rate used by the optimizer.
- epochs (int): The number of complete passes through the training dataset.
- results_dir (string): The directory path where the training results and model checkpoints will be saved.
- source_train_csv (string): The file path to the CSV containing training data. The CSV must have columns: 'filename', 'sex', and 'id'.
- source_val_csv (string): The file path to the CSV containing validation data. The CSV must have columns: 'filename', 'sex', and 'id'.
- source_test_csv (string): The file path to the CSV containing validation data. The CSV must have columns: 'filename', 'sex', and 'id'.
- saved_model_path: The file path to the trained model weights.
- verbose (boolean): Set to True for verbose logging.

