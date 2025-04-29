import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import random
from termcolor import colored
import warnings
from time import time
warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_elapsed_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Time Taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")



class MIMICCXRDataset(Dataset):
    def __init__(self, dataframe, image_col='path', label_cols='chexpert_labels', transform=None):
        self.dataframe = dataframe
        self.image_col = image_col
        self.transform = transform
        self.label_cols = label_cols
        
        # assert len(self.label_cols) == 14, f"Expected 14 label columns, got {len(self.label_cols)}"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.dataframe.iloc[idx][self.image_col]
            
        # Load and convert image
        # try:
        print(img_path)
        image = Image.open(img_path).convert('RGB')
        # except Exception as e:
        #     print(f"Error loading image {img_path}: {e}")
        #     # Return a placeholder image in case of error
        #     image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels (all 14 of them)
        # labels = self.dataframe.iloc[idx][self.label_cols].values.astype(np.float32)
        label_vals = np.array(list(self.dataframe.iloc[idx][self.label_cols].values()))
        labels = label_vals.astype(np.float32)
        labels = np.nan_to_num(labels)

        label_tensor = torch.tensor(labels, dtype=torch.float32)

        """
        Replace all instaces of -1 (denoting uncertainity for a condition) with 0
        """
        label_tensor = torch.where(label_tensor == -1, torch.tensor(0), label_tensor)        
        
        return image, label_tensor


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_classes=14, pretrained=True):
        super(MultiLabelClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features in the last layer
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            # Handle other model architectures
            in_features = self.model.get_classifier().in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

        if idx % 10 == 0:  # Print every 10 batches
            print(f"Batch {idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            all_outputs.append(outputs.sigmoid().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    return val_loss, all_outputs, all_labels


def calculate_metrics(outputs, labels, threshold=0.5):
    # Convert probabilities to binary predictions
    pred_labels = (outputs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': round(accuracy_score(labels.flatten(), pred_labels.flatten()), 3),
        'precision_micro': round(precision_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'recall_micro': round(recall_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'f1_micro': round(f1_score(labels, pred_labels, average='micro', zero_division=0), 3),
        'precision_macro': round(precision_score(labels, pred_labels, average='macro', zero_division=0), 3),
        'recall_macro': round(recall_score(labels, pred_labels, average='macro', zero_division=0), 3),
        'f1_macro': round(f1_score(labels, pred_labels, average='macro', zero_division=0), 3),
    }
    
    # Calculate AUC for each class and average
    try:
        aucs = []
        for i in range(labels.shape[1]):
            if len(np.unique(labels[:, i])) > 1:  # Only calculate AUC if there are both positive and negative samples
                aucs.append(roc_auc_score(labels[:, i], outputs[:, i]))
        
        if aucs:
            metrics['auc_macro'] = np.mean(aucs)
            metrics['auc_per_class'] = {i: auc for i, auc in enumerate(aucs)}
        else:
            metrics['auc_macro'] = float('nan')
    except:
        metrics['auc_macro'] = float('nan')
    
    return metrics

def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

def variable_size_collate_fn(batch):
    """
    Collate function for handling variable-sized images.
    Pads images to the maximum size in the batch.
    """
    # Filter out None values
    valid_samples = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]
    
    if len(valid_samples) == 0:
        return None, None
    
    images = []
    labels = []
    
    # Find max dimensions in this batch
    max_h = max([img.shape[1] for img, _ in valid_samples])
    max_w = max([img.shape[2] for img, _ in valid_samples])
    
    # Pad images to max dimensions
    for image, label in valid_samples:
        # Current image dimensions
        c, h, w = image.shape
        
        # Create new padded tensor
        padded_img = torch.zeros((c, max_h, max_w), dtype=image.dtype)
        padded_img[:, :h, :w] = image
        
        images.append(padded_img)
        
        # Handle any issues with labels
        if torch.isnan(label).any() or (label == -1).any():
            label = torch.nan_to_num(label, nan=0.0)
            label = torch.where(label == -1, torch.tensor(0.0), label)
            
        labels.append(label)
    
    # Stack the images and labels
    batched_images = torch.stack(images, dim=0)
    batched_labels = torch.stack(labels, dim=0)
    
    return batched_images, batched_labels

def main(args):
    seed_everything(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load MIMIC-CXR dataset from dataframe
    print("Loading MIMIC-CXR data...")
    
    # Load the dataframe containing paths and 14 labels
    df = pd.read_csv(args.csv_path)
    # df[args.image_col] = df[args.image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
    
    ##### Adding base image directory for real images
    assert 'img_type' in df.columns, "Column 'img_type' not found in DataFrame"

    # Check number of real and synthetic images
    num_real_images = len(df[df['img_type']=='real'])
    num_synthetic_images = len(df[df['img_type']=='synthetic'])

    print("Num Real Images: ", num_real_images)
    print("Num Synthetic Images: ", num_synthetic_images)

    if(num_real_images > 0):
        ##### Adding base image directory for real images
        df[args.image_col] = df[df['img_type']=='real'][args.image_col].apply(lambda x: os.path.join(args.real_image_dir, x))
    if(num_synthetic_images > 0):
        ##### Adding base image directory for synthetic images
        print("Synthetic image dir: ", args.synthetic_image_dir)
        df[args.image_col] = df[df['img_type']=='synthetic'][args.image_col].apply(lambda x: os.path.join(args.synthetic_image_dir, x))

    print(df['img_type'].value_counts())

    df['chexpert_labels'] = df['chexpert_labels'].apply(get_labels_dict_from_string)

    label_cols = list(df['chexpert_labels'].iloc[0].keys())

    # import pdb; pdb.set_trace()

    if(args.debug):
        df = df.sample(n=args.debug_samples, random_state=42).reset_index(drop=True)
    
    # Verify that the dataframe has the expected structure
    # expected_cols = 15  # 1 path column + 14 label columns
    # assert len(df.columns) == expected_cols, f"Expected {expected_cols} columns in dataframe, got {len(df.columns)}"
    
    # Print label distribution
    # print("Label distribution:")
    # label_cols = [col for col in df.columns if col != args.image_col]
    # for col in label_cols:
    #     positive_count = df[col].sum()
    #     print(f"  {col}: {positive_count} positive out of {len(df)} ({positive_count/len(df)*100:.2f}%)")
    
    # Split the data
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df[label_cols[0]] if args.stratify else None)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Create datasets
    train_dataset = MIMICCXRDataset(train_df, image_col=args.image_col, transform=train_transform)
    val_dataset = MIMICCXRDataset(val_df, image_col=args.image_col, transform=val_transform)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=variable_size_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=variable_size_collate_fn)
    
    # Create model
    num_classes = 14  # Fixed for MIMIC-CXR dataset
    model = MultiLabelClassifier(args.model_name, num_classes, pretrained=True)
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    print(colored(f"{model}", "green"))
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    train_losses = []
    val_losses = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f"{script_dir}/checkpoints/{args.model_name}", exist_ok=True)
    os.makedirs(f"{script_dir}/training_results", exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs...")
    # for epoch in tqdm(range(args.epochs), desc="Training Progress"):
    for epoch in range(args.epochs):
        # Train
        start_time = time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        # if(epoch + 1) % args.va_epochs == 0:
        val_loss, val_outputs, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
    
        # Calculate metrics
        metrics = calculate_metrics(val_outputs, val_labels, threshold=args.threshold)
        
        elapsed_time = time() - start_time
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        format_elapsed_time(elapsed_time)
        print(f"Metrics: ")
        for metric, value in metrics.items():
            if metric != 'auc_per_class':  # Don't print per-class AUC in the log
                print(f"  {metric}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': {k: v for k, v in metrics.items() if k != 'auc_per_class'},
            }, f"{script_dir}/checkpoints/{args.model_name}_{args.extra_info}.pth")
            print("Saved best model")
        
        # Update scheduler
        scheduler.step(val_loss)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{script_dir}/training_results/{args.model_name}_loss_curves.png")
    
    # Print final best metrics
    print("\nBest Validation Performance:")
    for metric, value in best_metrics.items():
        if metric != 'auc_per_class':  # Don't print per-class AUC in the final summary
            print(f"  {metric}: {value:.4f}")
            
    # Print per-class AUC scores
    if 'auc_per_class' in best_metrics:
        print("\nPer-class AUC scores:")
        for class_idx, auc in best_metrics['auc_per_class'].items():
            class_name = label_cols[class_idx]
            print(f"  {class_name}: {auc:.4f}")

    # Wriwte the best metrics to a CSV file
    results_df = pd.DataFrame([best_metrics])
    results_df.to_csv(f"{script_dir}/training_results/{args.model_name}_{args.extra_info}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label Classification on MIMIC-CXR")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name from timm")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--va_epochs", type=int, default=10, help="Number of epochs for validation")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--run", type=str, default="training", choices=['training', 'inference'], help="Run either training or inference.")

    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV with image paths and labels")
    parser.add_argument("--image_col", type=str, default="path", help="Column name in CSV that contains image paths")
    parser.add_argument("--real_image_dir", type=str, default=None, help="Base Directory containing images")
    parser.add_argument("--synthetic_image_dir", type=str, default=None, help="Base Directory containing synthetic images")

    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--stratify", action="store_true", help="Whether to stratify the train/val split")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode with a small subset of data")
    parser.add_argument("--debug_samples", type=int, default=500, help="Number of samples to use in debug mode")

    parser.add_argument("--extra_info", type=str, default=None, help="Extra info about an experiment") # Examples: real, mix, synthetic
    
    args = parser.parse_args()
    main(args)