import torch
import torch.nn as nn

class TransformerLossFunction(nn.Module):
    """
    A PyTorch module that computes the cross-entropy loss function for the Transformer model.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        """
        Initializes the TransformerLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification task.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predicted output and the actual output.

        Args:
        - predictions (torch.Tensor): The predicted output tensor.
        - targets (torch.Tensor): The actual output tensor.

        Returns:
        - torch.Tensor: The computed cross-entropy loss.
        """
        return self.criterion(predictions, targets)

    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predicted output and the actual output.

        Args:
        - predictions (torch.Tensor): The predicted output tensor.
        - targets (torch.Tensor): The actual output tensor.

        Returns:
        - torch.Tensor: The computed cross-entropy loss.
        """
        return self.forward(predictions, targets)


if __name__ == "__main__":
    # Create a dummy dataset
    predictions = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    # Initialize the loss function
    loss_fn = TransformerLossFunction(num_classes=5)

    # Compute the loss
    loss = loss_fn.get_loss(predictions, targets)

    print("Computed loss:", loss.item())