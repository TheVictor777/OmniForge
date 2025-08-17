from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 96.86%
    "Epochs": 25,
    "ModelPath": "BreedNova_Model.pth",
    "TrainFolder": "Datasets/70 Dog Breeds-Image Data Set/train",
    "TestFolder": "Datasets/70 Dog Breeds-Image Data Set/test",
    "LearningRate": 1e-7,
    "DatasetDownloadURL": "https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set",

    # Development settings - using a more successful model.
    "PreTrainedModel": "vit_b_16",
    "Dropout": 0.6,
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)
