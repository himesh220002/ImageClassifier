import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.amp import autocast, GradScaler # type: ignore
import time
import copy
from typing import Dict, Any

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.amp import autocast, GradScaler # type: ignore
import time
import copy
from typing import Dict, Any

class Trainer:
    """Orchestrates the training process with AMP, Early Stopping and LR Scheduling."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        patience: int = 5,
        scheduler = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        
        # Default scheduler if none provided
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)
        else:
            self.scheduler = scheduler
            
        self.scaler = GradScaler(device, enabled=(device == "cuda"))
        self.history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    def train_one_epoch(self, dataloader) -> float:
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(device_type=self.device, enabled=(self.device == "cuda")):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)

    def evaluate(self, dataloader) -> float:
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(device_type=self.device, enabled=(self.device == "cuda")):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
        return running_loss / len(dataloader.dataset)

    def train(self, train_loader, val_loader, num_epochs: int, epoch_callback = None) -> Dict[str, Any]:
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(current_lr)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {time.time()-start_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            if epoch_callback:
                epoch_callback(epoch + 1, num_epochs, train_loss, val_loss, time.time() - start_time)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.history
