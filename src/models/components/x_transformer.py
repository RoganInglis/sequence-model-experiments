import torch
from x_transformers import TransformerWrapper, Decoder
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XTransformer(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        super().__init__()

        self.model = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_length,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads
            ),
        ).to(device)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, mask=mask)


if __name__ == "__main__":
    _ = XTransformer()
