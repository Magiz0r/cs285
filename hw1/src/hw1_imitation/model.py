"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        self.hidden_dims = hidden_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        MLPnet = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim),
        )
        
        self.model = MLPnet.to(self.device)
        self.model = torch.compile(self.model)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor, # predicted actions
        action_chunk: torch.Tensor, # expert actions
    ) -> torch.Tensor:
        """ Compute MSE Loss """
        loss = self.criterion(state, action_chunk) # Compute loss between predicted and expert actions
            
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        B = state.shape[0] # batch size        
        actions = self.model(state) # FFN, predicted actions
        actions = actions.reshape(B, self.chunk_size, self.action_dim) # From [B, Chunk * actions] - > [B, C, A]. Because 1 chunk has 2 actions
                
        return actions


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        self.model = MLPmodel(state_dim, action_dim, chunk_size, hidden_dims)
        self.model = torch.compile(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_loss(
        self,
        state: torch.Tensor, 
        action_chunk: torch.Tensor, # Expert actions
    ) -> torch.Tensor:
        B = state.shape[0]
        
        x_0 = torch.randn(B, self.chunk_size, self.action_dim).to(self.device) # A_{t, 0} on the website
        tau = torch.rand(B, 1).to(self.device) # tau for NN to predict the velocity
        t = torch.rand(B, 1, 1).to(self.device) # same as tau, but adapted for computing A_{t, tau}
        x_t = t * action_chunk + (1 - t) * x_0 # A_{t, tau}
        velocity = self.model(x_t, state, tau) # v_theta(o, A, tau)
        velocity = velocity.reshape(B, self.chunk_size, self.action_dim)
        
        loss = ((velocity - (action_chunk - x_0)) ** 2).mean()
        
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        x = torch.randn(B, self.chunk_size, self.action_dim).to(self.device) # sample x_0 ~ N(0, I), from the notes
        dt = 1.0 / num_steps # delta t
        
        for step in range(num_steps): # integrate, for t in {0, delta t, 2 * delta t, ...}
            tau = torch.full( # for flow matching, to see the characteristic of "flow", we need to keep updated
                (B, 1),
                step / num_steps
            ).to(self.device)

            velocity = self.model(x, state, tau) # NN
            velocity = velocity.reshape(B, self.chunk_size, self.action_dim)
            x = x + dt * velocity # gradient ascent
            
        return x
        
    
    
class MLPmodel(nn.Module):
    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims):
        super().__init__()  
        self.hidden_dims = hidden_dims
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim
    
        self.net = nn.Sequential(
            nn.Linear(self.chunk_size * self.action_dim + self.state_dim + 1, self.hidden_dims[0]), # + 1 for time embedding
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[2], self.chunk_size * self.action_dim),
        )
        
        # self.model = self.net.to(self.device)
        # self.model = torch.compile(self.model)
        # # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.criterion = nn.MSELoss()
        
    def forward(self, x_t, state, step): # x current, state, time step
        B = state.shape[0]
        x_t = x_t.reshape(B, -1)
        tx = torch.cat([x_t, state, step], dim=-1)
        return self.net(tx)
        

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
