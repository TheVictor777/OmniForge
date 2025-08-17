from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 97.36%
    "Epochs": 25,
    "ModelPath": "Decksentience_Model.pth",
    "TrainFolder": "Datasets/Cards Image Dataset-Classification/train",
    "TestFolder": "Datasets/Cards Image Dataset-Classification/test",
    "LearningRate": 1e-5,
    "DatasetDownloadURL": "https://www.kaggle.com/datasets/86dcbfae1396038cba359d58e258915afd32de7845fd29ef6a06158f80d3cce8",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)
