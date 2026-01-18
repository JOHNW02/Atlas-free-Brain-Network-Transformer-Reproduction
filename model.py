import torch
import torch.nn as nn

class AtlasFreeBNT(nn.Module):
    def __init__(self,
                 roi_in_dim=1632,
                 roi_embed_dim=256,
                 K=3,
                 stride=2,
                 n_heads=4,
                 n_layers=1,
                 ff_mult=4,
                 dropout=0.2
                 ):
        super().__init__()
        self.roi_in_dim = roi_in_dim
        self.roi_embed_dim = roi_embed_dim
        self.K = K
        self.stride = stride
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.dropout = dropout
    

        # Define Feed-forward Module
        self.FNN = nn.Sequential(
            nn.Linear(self.roi_in_dim, self.roi_embed_dim),
            nn.GELU()
        )

        # Define Sum Block Pooling Module
        self.sumpool3d = nn.AvgPool3d(kernel_size=self.K, 
                                      stride=self.stride,
                                      divisor_override=1)

        # Define MHSA Transformer Module
        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=roi_embed_dim,
            nhead=self.n_heads,
            dim_feedforward=roi_embed_dim * ff_mult,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )        
        self.encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,
                                             num_layers=self.n_layers)

        # Define Classification Head                               
        self.out_norm = nn.LayerNorm(roi_embed_dim)
        self.classifier = nn.Linear(roi_embed_dim, 2)
        
    # Build Multi-channel Brain map
    # C: [B, X, Y, Z]
    # Q: [B, 400, roi_embed_dim]
    # return vol [B, X, Y, Z, roi_embed_dim]
    def brain_map(self, C, Q):

        B, X, Y, Z = C.shape
        L = C.numel() // B
        idx_flat = C.view(B, L)
        idx_flat = idx_flat.unsqueeze(-1).expand(B, L, Q.size(-1))

        out = Q.gather(1, idx_flat)
        out = out.view(B, X, Y, Z, Q.size(-1))

        return out

    def forward(self, C, F):

        Q = self.FNN(F)

        Brain_Map = self.brain_map(C, Q)

        Brain_Map = Brain_Map.permute(0, 4, 1, 2, 3).contiguous()

        pooled = self.sumpool3d(Brain_Map).flatten(2).transpose(1,2)

        X = self.encoder(pooled)

        X = self.out_norm(X)

        # Build subject-level feature vector using average pooling
        X = X.mean(dim=1)

        logits = self.classifier(X)
        return logits


