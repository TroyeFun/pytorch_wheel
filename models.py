import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class BaxterFK(BaseModel):

    def __init__(self, hidden_layers):
        layers = []

        input_size = 7
        for output_size in hidden_layers:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size

        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(input_size, 7)
        self.softmax = nn.Softmax(dim=1)
        self._init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        pos, ori_pow2 = x[:, :4], x[:, 4:]
        ori_pow2 = self.softmax(ori_pow2)  # w^2 + x^2 + y^2 + z^2 = 1
        return pos, ori_pow2

