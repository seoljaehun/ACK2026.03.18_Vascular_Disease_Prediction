import torch
import torch.nn as nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        
        self.eps = 1e-8 # log(0) 방지용 아주 작은 수

    def forward(self, m, d_prob):
        # 판별자 목표: 진짜(m=1)는 1로, 가짜(m=0)는 0으로
        # L_D = -[ m * log(D(x)) + (1-m) * log(1-D(x)) ]
        d_loss = -torch.mean(m * torch.log(d_prob + self.eps) + \
                             (1 - m) * torch.log(1. - d_prob + self.eps))
        return d_loss

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100.0):
        super(GeneratorLoss, self).__init__()
        
        self.alpha = alpha
        self.eps = 1e-8

    def forward(self, x, m, x_hat, d_prob):
        # 1. 생성자 목표 (속이기): 가짜(1-m)를 1로 예측하게 만듦
        # L_G_adv = - (1-m) * log(D(G(x)))
        adv_loss = -torch.mean((1 - m) * torch.log(d_prob + self.eps))
        
        # 2. 원본 유지 (MSE): 원본 데이터(m=1)는 똑같이 복원
        mse_numerator = torch.sum(m * ((x_hat - x) ** 2))
        mse_denominator = torch.sum(m) + 1e-8
        mse_loss = mse_numerator / mse_denominator
        
        # 최종 G_Loss
        g_loss = adv_loss + self.alpha * mse_loss
        
        return g_loss, adv_loss, mse_loss