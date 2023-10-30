import torch
import torch.nn as nn
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.act import ACTLayer
import sys
sys.path.append("/home/cx/")
from harl.models.base.attention_resnet_120 import Attention_model as Attention_model_resnet_120

class DuelingQNet(nn.Module):
    """Dueling Q Network for discrete action space."""

    def __init__(self, args, obs_space, output_dim, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        base_hidden_sizes = args["base_hidden_sizes"]
        base_activation_func = args["base_activation_func"]
        dueling_v_hidden_sizes = args["dueling_v_hidden_sizes"]
        dueling_v_activation_func = args["dueling_v_activation_func"]
        dueling_a_hidden_sizes = args["dueling_a_hidden_sizes"]
        dueling_a_activation_func = args["dueling_a_activation_func"]

        obs_shape = get_shape_from_obs_space(obs_space)
        print("!!!!!!!!!!!!! obs shape is ", obs_shape)
        # feature extractor
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape, base_hidden_sizes[0], base_activation_func
            )
            feature_dim = base_hidden_sizes[0]
        else:
            self.feature_extractor = Attention_model_resnet_120(args, obs_shape)
            feature_dim = obs_shape[0]

        # base
        base_sizes = [feature_dim] + list(base_hidden_sizes)
        self.base = PlainMLP(base_sizes, base_activation_func, base_activation_func)

        # dueling v
        dueling_v_sizes = [base_hidden_sizes[-1]] + list(dueling_v_hidden_sizes) + [1]
        print("dueling v sizes is ", dueling_v_sizes)
        self.dueling_v = PlainMLP(dueling_v_sizes, dueling_v_activation_func)

        # dueling a
        dueling_a_sizes = (
            [base_hidden_sizes[-1]] + list(dueling_a_hidden_sizes) + [output_dim]
        )
        self.dueling_a = PlainMLP(dueling_a_sizes, dueling_a_activation_func)

        self.to(device)

    def forward(self, obs):
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        print("x shape is ", x.shape)               
        x = self.base(x)  # (batch_size, )
        v = self.dueling_v(x) 
        a = self.dueling_a(x)
        print("a.shape", a.shape) # (batch_size, action_dim)
        return a - a.mean(dim=-1, keepdim=True) + v
