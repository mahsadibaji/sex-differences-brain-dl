# Sex Classification

## Running the training script
To start training the model, navigate to your project directory and run the run_train.py script with the required parameters. Below is the command and parameter explanations:
### Command
`python run_train.py
    --batch_size <batch_size>
    --learning_rate <learning_rate>
    --epochs <epochs> 
    --results_dir <path_to_results_dir> 
    --source_train_csv <path_to_train_csv> 
    --source_val_csv <path_to_val_csv> 
    --verbose <True/False>
`
### Arguments
- batch_size (int): The size of the batch used in training.
- learning_rate (float): The learning rate used by the optimizer.
- epochs (int): The number of complete passes through the training dataset.
- results_dir (string): The directory path where the training results and model checkpoints will be saved.
- source_train_csv (string): The file path to the CSV containing training data. The CSV should have columns: 'filename', 'sex', and 'id'.
- source_val_csv (string): The file path to the CSV containing validation data. The CSV should have columns: 'filename', 'sex', and 'id'.
- verbose (boolean): Set to True for verbose logging.
