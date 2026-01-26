import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import os
import pickle
import torch.nn.functional as F

# ================= [설정] =================
DATA_FILE = "/Users/sungminhong/Documents/deepleaning_proj/datasets/stgcn_dataset.pkl"
SAVE_PATH = "./checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"▶ 디바이스: {DEVICE}")

# ================= [데이터셋] =================
class SkeletonDataset(Dataset):
    def __init__(self, data_file):
        print(f"▶ 데이터 로드 중: {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            
        self.x_data = data['x_data'] # (N, 3, 300, 17, 1)
        self.y_data = data['y_data'] # (N,)
        
        # 클래스 정보 (레이블 기반 역추적은 어렵지만, 개수는 알 수 있음)
        self.num_classes = len(np.unique(self.y_data))
        print(f"▶ 데이터 로드 완료. 샘플 수: {self.x_data.shape[0]}, 클래스 수: {self.num_classes}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # 1. 데이터 가져오기 (3, 300, 17, 1)
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        # 2. 차원 정리 (3, 300, 17) - 마지막 M=1 차원 제거
        # 모델 입력: (C, T, V)
        x = x[..., 0] # (3, 300, 17)
        
        # 3. Tensor 변환
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()

# ================= [모델: ST-GCN] =================
class Graph:
    def __init__(self):
        self.num_node = 17
        # COCO Format Edges
        self.edges = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                      (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        self.A = self.get_A()
        
    def get_A(self):
        A = np.zeros((17, 17))
        for i, j in self.edges:
            A[i,j] = 1; A[j,i] = 1
        # Normalize: D^-0.5 * (A+I) * D^-0.5
        A = A + np.eye(17)
        D = np.sum(A, axis=0)
        D_inv = np.diag(D**(-0.5))
        D_inv[np.isinf(D_inv)] = 0
        adj = np.dot(np.dot(D_inv, A), D_inv)
        return torch.tensor(adj, dtype=torch.float32).to(DEVICE)

class GCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.A = A
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, (9,1), (stride,1), padding=(4,0)),
            nn.BatchNorm2d(out_c), nn.Dropout(0.5)
        )
        self.res = nn.Sequential(nn.Conv2d(in_c, out_c, 1, (stride,1)), nn.BatchNorm2d(out_c)) if in_c!=out_c or stride!=1 else lambda x:x
        
    def forward(self, x):
        res = self.res(x)
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, self.A) # Graph Conv
        return F.relu(self.tcn(x) + res)

class STGCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.graph = Graph()
        # Input: (N, C, T, V) where C=3 (x, y, confidence)
        # BatchNorm1d runs on (N, C*V, T) -> channels = 3*17 = 51
        self.bn = nn.BatchNorm1d(3 * 17) 
        
        self.layers = nn.ModuleList([
            GCNBlock(3, 64, self.graph.A), # In Channels = 3
            GCNBlock(64, 64, self.graph.A),
            GCNBlock(64, 128, self.graph.A, 2), # T dim reduction
            GCNBlock(128, 256, self.graph.A, 2)
        ])
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        N, C, T, V = x.size() # (N, 3, 300, 17)
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N,V,C,T).permute(0,2,3,1).contiguous() # (N, C, T, V)
        
        for l in self.layers: x = l(x)
        
        x = F.avg_pool2d(x, x.size()[2:]) # Global Pool
        return self.fc(x.view(N, -1))

# ================= [학습 루프] =================
def run():
    # 데이터 로드
    ds = SkeletonDataset(DATA_FILE)
    
    # Train/Val Split (8:2)
    train_len = int(0.8 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 정의
    model = STGCN(num_classes=ds.num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5) # LR 스케줄러 추가
    
    best_acc = 0.0
    
    print("\n🚀 학습 시작!")
    for ep in range(EPOCHS):
        model.train()
        loss_sum, corr, tot = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            _, pred = out.max(1)
            corr += pred.eq(y).sum().item()
            tot += y.size(0)
            
        scheduler.step()
        train_acc = 100 * corr / tot
        
        # 검증
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = out.max(1)
                v_corr += pred.eq(y).sum().item()
                v_tot += y.size(0)
        
        val_acc = 100 * v_corr / v_tot
        
        print(f"[Ep {ep+1}] Loss:{loss_sum/len(train_loader):.3f} | Tr_Acc:{train_acc:.1f}% | Val_Acc:{val_acc:.1f}% | LR:{opt.param_groups[0]['lr']:.5f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_model.pth"))
            print(f"    🌟 Best Model Saved! ({best_acc:.1f}%)")

    print(f"\n✅ 완료. 최종 Best Acc: {best_acc:.1f}%")

if __name__ == "__main__":
    run()