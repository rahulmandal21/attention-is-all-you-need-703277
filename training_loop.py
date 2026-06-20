import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

class TransformerTrainer:
    """
    A class used to train a Transformer model.

    Attributes:
    ----------
    model : nn.Module
        The Transformer model to be trained.
    optimizer : optim.Optimizer
        The optimizer used to update the model parameters.
    loss_fn : nn.Module
        The loss function used to compute the loss.
    device : torch.device
        The device used to train the model.
    warmup_steps : int
        The number of warmup steps.
    d_model : int
        The dimension of the model.
    max_grad_norm : float
        The maximum gradient norm.

    Methods:
    -------
    train_one_epoch(dataloader: torch.utils.data.DataLoader) -> float
        Trains the model for one epoch.
    validate(dataloader: torch.utils.data.DataLoader) -> float
        Validates the model on a given dataloader.
    should_stop_early(val_loss: float) -> bool
        Checks if the training should stop early.
    save_checkpoint(path: str) -> None
        Saves the model checkpoint.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device, warmup_steps: int, d_model: int, max_grad_norm: float = 1.0):
        """
        Initializes the TransformerTrainer.

        Args:
        ----
        model (nn.Module): The Transformer model to be trained.
        optimizer (optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (nn.Module): The loss function used to compute the loss.
        device (torch.device): The device used to train the model.
        warmup_steps (int): The number of warmup steps.
        d_model (int): The dimension of the model.
        max_grad_norm (float, optional): The maximum gradient norm. Defaults to 1.0.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.max_grad_norm = max_grad_norm
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Trains the model for one epoch.

        Args:
        ----
        dataloader (torch.utils.data.DataLoader): The dataloader for training.

        Returns:
        -------
        float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Validates the model on a given dataloader.

        Args:
        ----
        dataloader (torch.utils.data.DataLoader): The dataloader for validation.

        Returns:
        -------
        float: The average loss for the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.loss_fn(outputs, targets).item()
        return total_loss / len(dataloader)

    def should_stop_early(self, val_loss: float) -> bool:
        """
        Checks if the training should stop early.

        Args:
        ----
        val_loss (float): The validation loss.

        Returns:
        -------
        bool: True if the training should stop early, False otherwise.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        self.epochs_no_improve += 1
        return self.epochs_no_improve >= 5

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the model checkpoint.

        Args:
        ----
        path (str): The path to save the checkpoint.
        """
        torch.save(self.model.state_dict(), path)

    def get_lr(self, step_num: int) -> float:
        """
        Computes the learning rate.

        Args:
        ----
        step_num (int): The current step number.

        Returns:
        -------
        float: The learning rate.
        """
        return self.d_model**-0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)


if __name__ == "__main__":
    # Create a dummy model, optimizer, and loss function
    model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Create a dummy dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index: int):
            return torch.randn(10), torch.randn(10)

    dataloader = torch.utils.data.DataLoader(DummyDataset(100), batch_size=10)

    # Create a TransformerTrainer instance
    trainer = TransformerTrainer(model, optimizer, loss_fn, torch.device("cpu"), warmup_steps=1000, d_model=512)

    # Train the model for one epoch
    loss = trainer.train_one_epoch(dataloader)
    print(f"Loss: {loss}")

    # Validate the model
    val_loss = trainer.validate(dataloader)
    print(f"Validation Loss: {val_loss}")