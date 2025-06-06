import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random

# configuration
BASE_DIR = Path(__file__).parent.parent.resolve()
TEST_DIR = BASE_DIR / "data" / "processed" / "testing-1"
CLASSES = ['glioma', 'meningioma', 'pituitary']
MODEL_SAVE_PATH = BASE_DIR / "models" / "brain_tumor_model.pth"
GRAPHICS_DIR = BASE_DIR / "reports"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

# seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# dataset class
class TensorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.data = []
        self.labels = []

        print(f"preloading data from {self.root_dir}...")
        for label, cls in enumerate(CLASSES):
            cls_dir = self.root_dir / cls
            for file_path in sorted(cls_dir.glob("*.pt")):
                tensor = torch.load(file_path)
                self.data.append(tensor)
                self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# show one prediction per class
def show_sample_predictions(model, test_loader):
    model.eval()
    all_images = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_images.append(images.cpu())
            all_labels.append(labels)
            all_preds.append(preds.cpu())

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    found = [False] * len(CLASSES)
    selected_indices = []

    for i in range(len(all_labels)):
        true_class = all_labels[i].item()
        if not found[true_class]:
            selected_indices.append(i)
            found[true_class] = True
        if all(found):
            break

    n = len(selected_indices)
    plt.figure(figsize=(15, 5))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, i in enumerate(selected_indices):
        img = all_images[i].permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, n, idx + 1)
        plt.imshow(img)
        plt.title(f"Pred: {CLASSES[all_preds[i]]}\nTrue: {CLASSES[all_labels[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plot_path = GRAPHICS_DIR / "sample_predictions.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"sample predictions saved at: {plot_path}")

def main():
    print("loading test dataset...")
    test_dataset = TensorDataset(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    from model import get_efficientnet_v2_small

    print("building model...")
    model = get_efficientnet_v2_small(pretrained=False)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    show_sample_predictions(model, test_loader)

if __name__ == "__main__":
    main()