import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device =   torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class DeepARDataset(Dataset):
    def __init__(self, y, cat_id, dynamic_feat, context_len=30, pred_len=1):
        """
        y: shape [T]
        cat_id: int
        dynamic_feat: shape [num_feat, T]
        """
        self.y = y
        self.cat_id = cat_id
        self.dynamic_feat = dynamic_feat
        self.context_len = context_len
        self.pred_len = pred_len
        self.T = len(y)




    def __len__(self):
        return self.T - self.context_len - self.pred_len

    def __getitem__(self, idx):
        past_y = torch.tensor(self.y[idx:idx+self.context_len], dtype=torch.float32)  # [T]
        past_cov = torch.tensor(self.dynamic_feat[:, idx:idx+self.context_len], dtype=torch.float32)  # [F,T]
        target = torch.tensor(self.y[idx+self.context_len:idx+self.context_len+self.pred_len], dtype=torch.float32)

        return past_y, past_cov, torch.tensor(self.cat_id), target

class DeepAR(nn.Module):
    def __init__(self, context_len, future_steps, num_dyn_feat, cat_cardinality, cat_emb_dim=4,
                 hidden_size=64, num_layers=1):
        super().__init__()


        self.cat_emb = nn.Embedding(cat_cardinality, cat_emb_dim)
        input_size = 1 + num_dyn_feat + cat_emb_dim
        self.context_len = context_len
        self.future_steps = future_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_size, 1)


    def forward(self, past_y, past_cov, cat_id):
        """
        past_y: [B,T]
        past_cov: [B,features,T]
        cat_id: [B]
        """
        B, T = past_y.shape
        F = past_cov.shape[1]

        cat_e = self.cat_emb(cat_id)          # [B,emb]
        cat_e = cat_e.unsqueeze(1).repeat(1,T,1)

        y_in = past_y.unsqueeze(-1)           # [B,T,1]
        cov = past_cov.permute(0,2,1)         # [B,T,F]

        x = torch.cat([y_in, cov, cat_e], dim=-1)  # [B,T,1+F+emb]

        out, _ = self.lstm(x)
        pred = self.proj(out[:, -1, :])       # 取最后一步

        return pred.squeeze(-1)

def model_train(model, y, cat_id, dynamic_feat, context_len, epochs, lr):


    iteration = 0
    while True:
        hasNan = False
        iteration += 1

        dynamic_feat_full = np.stack([dynamic_feat])  # [1, T+future_steps]

        ds = DeepARDataset(y, cat_id, dynamic_feat_full, context_len=context_len, pred_len=1)
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total = 0
            for past_y, past_cov, cid, target in dl:
                past_y = past_y.to(device)
                past_cov = past_cov.to(device)
                cid = cid.to(device)
                target = target.to(device)

                pred = model(past_y, past_cov, cid)
                loss = loss_fn(pred, target.squeeze(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()

            if math.isnan(total):
                hasNan = True
                break
                print("Training encounter Nan at epoch ", epoch, " in iteration ", iteration)

        if hasNan == False:
            break
        # print(f"epoch {epoch + 1}: loss={total / len(dl):.4f}")

def forecast(model, y, cat_id, dynamic_feat_full, context_len, future_steps):
    """
    dynamic_feat_full: shape [F, T + future_steps]
    """
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    preds = []

    dynamic_feat_full = np.stack([dynamic_feat_full])

    y_list = list(y)  # 会不断 append 预测值
    T_hist = len(y)  # 原始历史长度

    for step in range(future_steps):
        t = T_hist + step  # 当前要预测的时间索引（未来）

        start = t - context_len
        end = t

        # y 部分：完全来自 y_list（历史 + 之前预测）
        past_y = torch.tensor(
            y_list[start:end], dtype=torch.float32
        ).unsqueeze(0).to(device)

        # cov 部分：从 dynamic_feat_full 对齐切片
        past_cov_np = dynamic_feat_full[:, start:end]  # 保证这里也是 context_len
        past_cov = torch.tensor(
            past_cov_np, dtype=torch.float32
        ).unsqueeze(0).to(device)

        cid = torch.tensor([cat_id]).to(device)

        with torch.no_grad():
            pred = model(past_y, past_cov, cid).item()

        preds.append(pred)
        y_list.append(pred)

    return preds