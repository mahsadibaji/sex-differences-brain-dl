# Sex Classification

## Training the Model
To start training the model, navigate to your project directory and run the run_train.py script with the required parameters. Below is the command and parameter explanations:
### Command
```
python /scripts/run_train.py --batch_size 16 --learning_rate 0.01 --epochs 50 --results_dir <path_to_results_dir> --source_train_csv /data/train.csv --source_val_csv /data/valid.csv --verbose <True/False>
```
### Arguments
- batch_size (int): The size of the batch used in training.
- learning_rate (float): The learning rate used by the optimizer.
- epochs (int): The number of complete passes through the training dataset.
- results_dir (string): The directory path where the training results and model checkpoints will be saved.
- source_train_csv (string): The file path to the CSV containing training data. The CSV must have columns: 'filename', 'sex', and 'id'.
- source_val_csv (string): The file path to the CSV containing validation data. The CSV must have columns: 'filename', 'sex', and 'id'.
- verbose (boolean): Set to True for verbose logging.

## Testing the model

### Command
```
python /scripts/run_test.py --results_dir <path_to_results_dir> --source_test_csv /data/test_data.csv --saved_model_path /pretrained-weights/sfcn_best.pth --verbose <True/False>
```
