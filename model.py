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
    

        self.FNN = nn.Sequential(
            nn.Linear(self.roi_in_dim, self.roi_embed_dim),
            nn.GELU()
        )

        self.sumpool3d = nn.AvgPool3d(kernel_size=self.K, 
                                      stride=self.stride,
                                      divisor_override=1)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=roi_embed_dim,
            nhead=self.n_heads,
            dim_feedforward=roi_embed_dim * ff_mult,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            #norm_first=True,
        )        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, roi_embed_dim))
        # initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.enc_layer,
                                             num_layers=self.n_layers)
                                             
        self.out_norm = nn.LayerNorm(roi_embed_dim)
        self.classifier = nn.Linear(roi_embed_dim, 2)
        
    
    def brain_map(self, C, Q):
        # C: [B, X, Y, Z]
        # Q: [B, 400, 256]
        # return vol [B, X, Y, Z, 256]

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

        cls_token = self.cls_token.expand(pooled.size(0), -1, -1)      # [B, 1, V]
        tokens = torch.cat([cls_token, pooled], dim=1)   # [B, 1+U, V]

        X = self.encoder(tokens)

        X = self.out_norm(X)

        # subject-level feature vector
        #X = X.mean(dim=1)
        X_cls = X[:, 0, :]

        logits = self.classifier(X_cls)
        #logits = self.classifier(X)

        return logits


