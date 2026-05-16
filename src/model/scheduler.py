import math
import torch


class LinearAlphaScheduler:
    def alpha(self, t):            
      return 1.0 - t

    def alpha_derivative(self, t): 
      return -torch.ones_like(t)

    def weight(self, t):           
      return -self.alpha_derivative(t) / (1 - self.alpha(t) + 1e-6)

    def reverse_mask_prob(self, s, t): 
      return (1 - self.alpha(s)) / (1 - self.alpha(t) + 1e-6)


class CosineAlphaScheduler:
    def alpha(self, t):            
      return 1 - torch.cos((math.pi / 2) * (1 - t))

    def alpha_derivative(self, t): 
      return -(math.pi / 2) * torch.sin((math.pi / 2) * (1 - t))

    def weight(self, t):           
      return -self.alpha_derivative(t) / (1 - self.alpha(t) + 1e-6)

    def reverse_mask_prob(self, s, t): 
      return (1 - self.alpha(s)) / (1 - self.alpha(t) + 1e-6)


if __name__ == "__main__":
    