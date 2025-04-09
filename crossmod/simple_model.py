import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, device):
        super(SimpleModel, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(device)
        
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
        ).to(device)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids).to(dtype=torch.float32)
        encoded = self.encoder(embedded).to(dtype=torch.float32)
        encoded = encoded * attention_mask.unsqueeze(-1).to(torch.float32)
        return encoded
