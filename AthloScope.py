from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 97.20%
    "Epochs": 25,
    "ModelPath": "AthloScope_Model.pth",
    "TrainFolder": "Datasets/100 Sports Image Classification/train",
    "TestFolder": "Datasets/100 Sports Image Classification/test",
    "LearningRate": 1e-5,
    "DatasetDownloadURL": "https://www.kaggle.com/datasets/gpiosenka/sports-classification",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)
