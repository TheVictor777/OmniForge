"""
This program creates a custom framework and pipeline that are used in various AI image classification projects.
It is designed to be flexible and efficient, allowing for easy integration of different models and datasets.
"""
from utils import load_model, model_to_multi_gpu, set_random_seed  # Importing efficiency utility functions
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchinfo
import torch
import tqdm
import os

class classiforge_dataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images from a directory structure where each subdirectory represents a class.

    Args:
        MainDirectory (str): The main directory containing class subdirectories.
    Returns:
        torch.utils.data.Dataset: A dataset object for loading images and their corresponding labels.
    """
    def __init__(self, MainDirectory: str):
        self.MainDirectory = MainDirectory
        self.ClassFolders = sorted([Directory for Directory in os.listdir(MainDirectory) if os.path.isdir(os.path.join(MainDirectory, Directory))])  # Sort class folders.
        self.ImagePaths = []
        self.Labels = []

        for Index, class_name in enumerate(self.ClassFolders):
            ClassPaths = os.path.join(MainDirectory, class_name)
            for ImageFiles in os.listdir(ClassPaths):
                if ImageFiles.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.ImagePaths.append(os.path.join(ClassPaths, ImageFiles))
                    self.Labels.append(Index)

        self.DatasetTransform = transforms.Compose([
            transforms.Resize((224, 224)),  # Scale.
            transforms.RandomHorizontalFlip(),  # Horizontal flip.
            transforms.RandomVerticalFlip(),  # Vertical flip.
            transforms.ToTensor(),

            # Normalize for resnet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ImagePaths)

    def __getitem__(self, Index):
        Photo = Image.open(self.ImagePaths[Index]).convert("RGB")
        Photo = self.DatasetTransform(Photo)
        Label = torch.tensor(self.Labels[Index], dtype=torch.long)
        return Photo, Label

def train_step(Model: torch.nn.Module, dataLoader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Train the model for one epoch.

    Args:
        Model (torch.nn.Module): The model to train.
        dataLoader (torch.utils.data.DataLoader): The data loader for training data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the training on.
    """
    Model.train()
    RunningLoss = 0.0
    for Index, (InputTensor, Labels) in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader), desc="Training"):
        InputTensor, Labels = InputTensor.to(device), Labels.to(device)
        optimizer.zero_grad()
        OutputTensor = Model(InputTensor)
        Loss = criterion(OutputTensor, Labels)
        Loss.backward()
        optimizer.step()
        RunningLoss += Loss.item()
    print(f"[CLASSIFORGE] Epoch training loop finished with training loss: {RunningLoss/len(dataLoader):.7f}")

def test_step(Model: torch.nn.Module, dataLoader: torch.utils.data.DataLoader, device: torch.device):
    """
    Test the model and calculate accuracy.

    Args:
        Model (torch.nn.Module): The model to test.
        dataLoader (torch.utils.data.DataLoader): The data loader for testing data.
        device (torch.device): The device to run the testing on.
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    Model.eval()
    Correct, Total = 0, 0
    with torch.no_grad():
        for Index, (InputTensor, Labels) in tqdm.tqdm(enumerate(dataLoader), total=len(dataLoader), desc="Testing"):
            InputTensor, Labels = InputTensor.to(device), Labels.to(device)
            OutputTensor = Model(InputTensor)
            _, Preds = torch.max(OutputTensor, 1)
            Correct += (Preds == Labels).sum().item()
            Total += Labels.size(0)
    Accuracy = Correct/Total*100
    print(f"[CLASSIFORGE] Epoch testing loop finished with testing accuracy: {Accuracy:.2f}%")
    return Accuracy

def validate_dataset(TrainFolder: str, TestFolder: str, DatasetDownloadURL: str = None):
    """
    Validate the dataset by checking if the training and testing folders exist and contain data files.

    Args:
        TrainFolder (str): The path to the training folder.
        TestFolder (str): The path to the testing folder.
        DatasetDownloadURL (str): Optional URL for downloading the dataset if it does not exist.
    """
    # Check training folder validity.
    if not os.path.isdir(TrainFolder):
        print("\033[91m" + f"Training folder '{TrainFolder}' does not exist." + "\033[0m")  # RED
        if DatasetDownloadURL: print("\033[93m" + f"Please download the dataset from {DatasetDownloadURL} before trying again." + "\033[0m")  # YELLOW
        quit()

    # Check testing folder validity.
    if not os.path.isdir(TestFolder):
        print("\033[91m" + f"Testing folder '{TestFolder}' does not exist." + "\033[0m")  # RED
        if DatasetDownloadURL: print("\033[93m" + f"Please download the dataset from {DatasetDownloadURL} before trying again." + "\033[0m")  # YELLOW
        quit()

    # Check if there are files in the training folder.
    if len(os.listdir(TrainFolder)) == 0: raise ValueError(f"Training folder '{TrainFolder}' is empty.")

    # Check if there are files in the testing folder.
    if len(os.listdir(TestFolder)) == 0: raise ValueError(f"Testing folder '{TestFolder}' is empty.")

    print("\033[92m" + "[CLASSIFORGE] Dataset validation passed." + "\033[0m")  # GREEN

def start_training(Epochs: int, ModelPath: str, TrainFolder: str, TestFolder: str, LearningRate: float = 1e-4, BatchSize: int = 64, DatasetDownloadURL: str = None, PreTrainedModel: str = "resnet18", Dropout: float = 0.5, WeightDecay: float = 1e-6, Num_Workers = max(1, os.cpu_count()-1)):
    """
    Start the training process for the model.

    Args:
        Epochs (int): The number of epochs to train the model.
        ModelPath (str): The path where the model will be saved.
        TrainFolder (str): The path to the training dataset folder.
        TestFolder (str): The path to the testing dataset folder.
        LearningRate (float): The learning rate for the optimizer.
        BatchSize (int): The batch size for training and testing.
        DatasetDownloadURL (str): Optional URL for downloading the dataset if it does not exist.
        PreTrainedModel (str): The pre-trained model to use, e.g., "resnet18" or "vit_b_16".
        Dropout (float): Dropout rate for the model's fully connected layer.
        WeightDecay (float): Weight decay for the optimizer.
        Num_Workers (int): Number of workers for data loading, set to one less than the total CPU cores available.
    """

    # Set random seed.
    set_random_seed(42)

    # Immediately check dataset validity.
    validate_dataset(TrainFolder, TestFolder, DatasetDownloadURL)

    # Set up distributed training if applicable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\033[95m" + f"[CLASSIFORGE] Using device: {device}" + "\033[0m")  # PINK

    # Set up datasets.
    train_dataset = classiforge_dataset(TrainFolder)
    test_dataset = classiforge_dataset(TestFolder)

    # Setup model.
    if PreTrainedModel == "resnet18":
        Backbone = models.resnet18(weights='IMAGENET1K_V1')
        Backbone.fc = nn.Sequential(nn.Dropout(p=Dropout), nn.Linear(Backbone.fc.in_features, len(train_dataset.ClassFolders)))
    elif PreTrainedModel == "vit_b_16":
        Backbone = models.vit_b_16(weights='IMAGENET1K_V1')
        Backbone.heads = nn.Sequential(nn.Dropout(p=Dropout), nn.Linear(Backbone.heads.head.in_features, len(train_dataset.ClassFolders)))
    else:
        raise ValueError(f"Unsupported model: {PreTrainedModel}.")

    Model = Backbone.to(device)
    Model, HighestAccuracy = load_model(Model, ModelPath)
    Model = model_to_multi_gpu(Model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=Num_Workers, prefetch_factor=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=Num_Workers, prefetch_factor=1)

    # Get model summary only on the main process to avoid clutter.
    print("\033[94m" + "[CLASSIFORGE] Model Summary:" + "\033[0m")  # BLUE
    torchinfo.summary(Model, input_size=(1, 3, 224, 224), device=device.type)  # A single batched RGB image of size 224x224.

    # Training Parameters.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(Model.parameters(), lr=LearningRate, weight_decay=WeightDecay)

    # Training Loop.
    for Epoch in range(Epochs):
        print("\033[95m" + f"[CLASSIFORGE] Epoch {Epoch+1}/{Epochs}" + "\033[0m")  # PINK

        train_step(Model, train_loader, criterion, optimizer, device)
        Accuracy = test_step(Model, test_loader, device)

        # Save the model only from the main process.
        if Accuracy > HighestAccuracy:
            HighestAccuracy = Accuracy
            # Correctly save the model's state_dict when wrapped.
            save_data = {
                'model_state_dict': Model.state_dict() if not isinstance(Model, torch.nn.DataParallel) else Model.module.state_dict(),
                'highest_accuracy': HighestAccuracy
            }
            torch.save(save_data, ModelPath)
            print("\033[92m" + "[CLASSIFORGE] Best model saved!" + "\033[0m")
        print()  # Separator for clarity between epochs.

if __name__ == "__main__":
    print("This is a utility module and is not meant to be run directly.")
    print("Please import this module in your project to use the ClassiForge framework.")
    # Example usage:
    # start_training(Epochs=10, ModelPath='model.pth', TrainFolder='path/to/train', TestFolder='path/to/test')
