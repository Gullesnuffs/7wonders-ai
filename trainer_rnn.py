import torch


class TrainerRNN:
    def __init__(self, optimizer, device):
        self.optimizer = optimizer
        self.device = device
        self.reset()

    def reset(self):
        self.loss_counter = 0
        self.steps = 0

    def backprop(self, loss: torch.tensor):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.steps += 1

    # def total_loss(self) -> float:
        # return self.total_loss / max(1, self.steps)
