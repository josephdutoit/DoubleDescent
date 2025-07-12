import torch.nn
import torch
from typing import Callable

class LinearRegressionModel(torch.nn.Module):
    def __init__(
            self, 
            num_samples: int,
            feature_dim: int,
            make_design_matrix: Callable | None = None,
            ) -> None:
        super(LinearRegressionModel, self).__init__()
        
        if make_design_matrix is None:
            self.make_design_matrix = lambda x: torch.cat(
                (torch.ones(x.size(0), 1).to(x.device), x[:num_samples, :]), dim=1
            )
        else:
            self.make_design_matrix = make_design_matrix
        self.M_tm = None

        self.linear = torch.nn.Linear(feature_dim, 1, bias=False)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.make_design_matrix(x)

        self.M_tm = x

        # print("M_tm shape:", self.M_tm.shape)
        # print("M_tm: ", self.M_tm)
        return self.linear(x)

    def get_aliasing_metrics(self) -> tuple:
        M_tm_inv = torch.linalg.pinv(self.M_tm)
        aliasing_norm = torch.norm(M_tm_inv, p=2)
        B = M_tm_inv @ self.M_tm
        data_insufficiency = torch.norm(B - torch.eye(B.size(0)).to(B.device), p=2)
        return aliasing_norm, data_insufficiency



