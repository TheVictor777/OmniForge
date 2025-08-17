from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 98.40%
    "Epochs": 25,
    "ModelPath": "FlutterFrame_Model.pth",
    "TrainFolder": "Datasets/Butterfly & Moths Image Classification 100 species/train",
    "TestFolder": "Datasets/Butterfly & Moths Image Classification 100 species/test",
    "LearningRate": 1e-5,
    "DatasetDownloadURL": "https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)
