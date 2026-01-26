import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# GUI에서 실행할 때도 디바이스 설정이 필요합니다.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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
        # DEVICE 변수를 사용하므로 상단에 정의가 필요함
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
        self.bn = nn.BatchNorm1d(3 * 17) 
        
        self.layers = nn.ModuleList([
            GCNBlock(3, 64, self.graph.A), # In Channels = 3
            GCNBlock(64, 64, self.graph.A),
            GCNBlock(64, 128, self.graph.A, 2), # T dim reduction
            GCNBlock(128, 256, self.graph.A, 2)
        ])
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        N, C, T, V = x.size() 
        # (N, 3, T, 17) -> 변환
        x = x.permute(0,3,1,2).contiguous().view(N, V*C, T)
        x = self.bn(x)
        x = x.view(N,V,C,T).permute(0,2,3,1).contiguous() 
        
        for l in self.layers: x = l(x)
        
        # Global Avg Pool
        x = F.avg_pool2d(x, x.size()[2:]) 
        return self.fc(x.view(N, -1))