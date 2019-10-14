import torch
import random

class DymanicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DymanicNet,self).__init__()
        self.input_layer = torch.nn.Linear(D_in, H)
        self.middle_layer = torch.nn.Linear(H, H)
        self.output_layer = torch.nn.Linear(H, D_out)

    def forward(self,x):
        h_relu = self.input_layer(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_layer(h_relu).clamp(min=0)
        y_pred = self.output_layer(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DymanicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 ==99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

