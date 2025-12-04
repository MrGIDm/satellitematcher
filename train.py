import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import kornia as K
import kornia.feature as KF



class SatellitePairDataset(Dataset):

    def __init__(self, data_dir, max_size=640):
        self.max_size = max_size
        self.pairs = []

        # collecting all image pairs from all cities
        data_path = Path(data_dir)
        cities = [d for d in data_path.iterdir() if d.is_dir()]

        for city_dir in cities:
            img1_path = city_dir / "pair" / "img1.png"
            img2_path = city_dir / "pair" / "img2.png"

            if img1_path.exists() and img2_path.exists():
                self.pairs.append({
                    'img1': str(img1_path),
                    'img2': str(img2_path),
                    'city': city_dir.name
                })

        print(f"Loaded {len(self.pairs)} image pairs from {len(cities)} cities")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # load and convert to grayscale
        img1 = cv2.imread(pair['img1'], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(pair['img2'], cv2.IMREAD_GRAYSCALE)

        # resize if needed
        h1, w1 = img1.shape
        if max(h1, w1) > self.max_size:
            scale = self.max_size / max(h1, w1)
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)
            img2 = cv2.resize(img2, None, fx=scale, fy=scale)

        # to tensor [1, H, W] and normalize
        img1_t = torch.from_numpy(img1).float().unsqueeze(0) / 255.
        img2_t = torch.from_numpy(img2).float().unsqueeze(0) / 255.

        return {
            'image0': img1_t,
            'image1': img2_t,
            'pair_name': pair['city']
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        img0 = batch['image0'].to(device)
        img1 = batch['image1'].to(device)

        optimizer.zero_grad()

        try:
            correspondences = model({
                'image0': img0,
                'image1': img1
            })

            confidence = correspondences['confidence']

            if len(confidence) == 0:
                pbar.set_postfix({'loss': 'no matches', 'matches': 0})
                continue


            conf_loss = -torch.mean(confidence)

            # regularization, don't produce too many or too few matches
            num_matches = len(confidence)
            target = 200.0
            match_reg = 0.0001 * (num_matches - target) ** 2 / target

            total_loss_batch = conf_loss + match_reg

            total_loss_batch.backward()

            # clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += total_loss_batch.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'matches': num_matches
            })

        except Exception as e:
            print(f"\nError in batch: {e}")
            continue

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device):
    model.eval()
    total_matches = 0
    total_confidence = 0

    with torch.no_grad():
        for batch in dataloader:
            img0 = batch['image0'].to(device)
            img1 = batch['image1'].to(device)

            correspondences = model({
                'image0': img0,
                'image1': img1
            })

            total_matches += len(correspondences['confidence'])
            total_confidence += torch.mean(correspondences['confidence']).item()

    avg_matches = total_matches / len(dataloader)
    avg_conf = total_confidence / len(dataloader)

    return avg_matches, avg_conf


def main():
    print("\n" + "="*60)
    print("LoFTR Fine-tuning on ONERA Dataset")
    print("="*60 + "\n")

    # cfg
    DATA_DIR = Path(r"data\onera-dataset")
    BATCH_SIZE = 1  # LoFTR is memory-intensive
    EPOCHS = 10
    LR = 1e-5  # small learning rate for fine-tuning
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}\n")

    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please update DATA_DIR path in the script")
        return

    # load pretrained model
    print("Loading pretrained LoFTR...")
    model = KF.LoFTR(pretrained='outdoor').to(DEVICE)
    print("Loaded\n")

    dataset = SatellitePairDataset(DATA_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train pairs: {len(train_dataset)}")
    print(f"Val pairs: {len(val_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for param in model.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.Adam(trainable_params, lr=LR)

    best_matches = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_matches, val_conf = validate(model, val_loader, DEVICE)

        print(f"\nResults:")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val matches: {val_matches:.1f}")
        print(f"Val confidence: {val_conf:.3f}")

        # save best model
        if val_matches > best_matches:
            best_matches = val_matches
            torch.save(model.state_dict(), 'loftr_finetuned_best.pth')
            print(f"Saved best model (matches: {val_matches:.1f})")

    # save final model
    torch.save(model.state_dict(), 'loftr_finetuned_final.pth')

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation matches: {best_matches:.1f}")
    print("Saved: loftr_finetuned_best.pth")
    print("="*60)


if __name__ == "__main__":
    main()