import torch
import torch.nn as nn
import torch.nn.functional as F

class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_r = nn.Linear(in_features, out_features)
        self.w_x = nn.Linear(in_features, out_features)
        self.w_y = nn.Linear(in_features, out_features)
        self.w_z = nn.Linear(in_features, out_features)

    def forward(self, r, x, y, z):
        r_ = self.w_r(r) - self.w_x(x) - self.w_y(y) - self.w_z(z)
        x_ = self.w_r(x) + self.w_x(r) + self.w_y(z) - self.w_z(y)
        y_ = self.w_r(y) - self.w_x(z) + self.w_y(r) + self.w_z(x)
        z_ = self.w_r(z) + self.w_x(y) - self.w_y(x) + self.w_z(r)
        return r_, x_, y_, z_

class QuaternionNet(nn.Module):
    def __init__(self, input=7, hidden_dim=128, output=7):
        super().__init__()
        self.in_dim = input 
        self.out_dim = output 
        
        # 3 hidden quaternion layers
        self.q1 = QuaternionLinear(self.in_dim, hidden_dim)
        self.q2 = QuaternionLinear(hidden_dim, hidden_dim)
        self.q3 = QuaternionLinear(hidden_dim, self.out_dim)

    def forward(self, x):
        # x: [B, 28] → split into (r, x, y, z)
        B = x.shape[0]
        x = x.view(B, -1, 4)  # [B, 7, 4]
   
        r, i, j, k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        r = r.reshape(B, -1)
        i = i.reshape(B, -1)
        j = j.reshape(B, -1)
        k = k.reshape(B, -1)

        # Layer 1
        r, i, j, k = self.q1(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        # Layer 2
        r, i, j, k = self.q2(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        # Layer 3
        r, i, j, k = self.q3(r, i, j, k)

        # Output: [B, 7 * 4]
        out = torch.stack([r, i, j, k], dim=-1)  # [B, 7, 4]
        
        out = out.view(B, 1,-1)
        
        return out

class QVNN_AutoEncoder(nn.Module):
    def __init__(self, in_quats=7, out_quats=7, latent_dim=8):
        super().__init__()
        self.in_quats = in_quats
        self.out_quats = out_quats

        # Encoder
        self.enc1 = QuaternionLinear(in_quats, 128)
        self.enc2 = QuaternionLinear(128, 64)
        self.enc3 = QuaternionLinear(64, latent_dim)

        # Decoder
        self.dec1 = QuaternionLinear(latent_dim, 64)
        self.dec2 = QuaternionLinear(64, 128)
        self.dec3 = QuaternionLinear(128, out_quats)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.in_quats, 4)
        r, i, j, k = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        # Encoder
        r, i, j, k = self.enc1(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        r, i, j, k = self.enc2(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        r, i, j, k = self.enc3(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        # Decoder
        r, i, j, k = self.dec1(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        r, i, j, k = self.dec2(r, i, j, k)
        r, i, j, k = F.relu(r), F.relu(i), F.relu(j), F.relu(k)

        r, i, j, k = self.dec3(r, i, j, k)

        out = torch.stack([r, i, j, k], dim=-1)  # [B, 7, 4]
        
        out = out.view(B, 1,-1) # [B,1,28]
        return out
