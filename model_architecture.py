import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    A PyTorch implementation of the Transformer model architecture.
    
    The Transformer model consists of an encoder and a decoder, each composed of a stack of identical layers.
    The encoder maps an input sequence to a sequence of continuous representations, and the decoder generates an output sequence one element at a time.
    The model uses multi-head attention, self-attention, and position-wise feed-forward networks to compute representations of its input and output.
    """
    
    def __init__(self, input_dim: int, output_dim: int, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        """
        Initializes the Transformer model.
        
        Args:
        input_dim (int): The dimension of the input sequence.
        output_dim (int): The dimension of the output sequence.
        d_model (int): The dimension of the model.
        num_heads (int): The number of heads in the multi-head attention.
        num_layers (int): The number of layers in the encoder and decoder.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model, dropout=dropout)
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model, dropout=dropout)
        self.encoder_stack = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.decoder_stack = nn.TransformerDecoder(self.decoder, num_layers=num_layers)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(d_model, output_dim)
        
    def forward(self, input_seq: torch.Tensor, output_seq: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Transformer model.
        
        Args:
        input_seq (torch.Tensor): The input sequence.
        output_seq (torch.Tensor): The output sequence.
        
        Returns:
        torch.Tensor: The output of the model.
        """
        input_embedding = self.input_embedding(input_seq)
        output_embedding = self.output_embedding(output_seq)
        encoder_output = self.encoder_stack(input_embedding)
        decoder_output = self.decoder_stack(output_embedding, encoder_output)
        return decoder_output
    
    def count_parameters(self) -> int:
        """
        Counts the number of parameters in the model.
        
        Returns:
        int: The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerModel(input_dim=512, output_dim=512, d_model=512, num_heads=8, num_layers=6)
    input_seq = torch.randn(1, 10, 512)
    output_seq = torch.randn(1, 10, 512)
    output = model(input_seq, output_seq)
    print(output.shape)