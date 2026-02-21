import torch
import torch.optim as optim
from utils.Config import Config 
from tqdm import tqdm

class GAINTrainer:
    def __init__(self, generator, discriminator, criterion_G, criterion_D, hint_rate=0.9):
        self.device = Config.device
        self.hint_rate = hint_rate
        
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        
        self.criterion_G = criterion_G.to(self.device)
        self.criterion_D = criterion_D.to(self.device)
        
        # 표준 Adam 옵티마이저
        self.opt_G = optim.Adam(self.G.parameters(), lr=Config.learning_rate)
        self.opt_D = optim.Adam(self.D.parameters(), lr=Config.learning_rate)

    def train_epoch(self, train_loader):
        self.G.train()
        self.D.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_mse_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Train", leave=False)
        
        for batch_x, batch_m in pbar:
            batch_x = batch_x.to(self.device)
            batch_m = batch_m.to(self.device)
            
            # 1. 노이즈 및 힌트 생성
            z = torch.empty_like(batch_x).uniform_(0, 0.01).to(self.device)
            x_tilde = batch_m * batch_x + (1 - batch_m) * z # 빈칸을 노이즈로 채움
            
            h_temp = (torch.rand_like(batch_m) < self.hint_rate).float()
            hint_m = batch_m * h_temp + 0.5 * (1 - h_temp)
            
            # ==========================
            # 2. 판별자(D) 학습
            # ==========================
            self.opt_D.zero_grad()
            
            x_hat = self.G(x_tilde, batch_m)
            x_bar = batch_m * batch_x + (1 - batch_m) * x_hat # 복원된 데이터
            
            d_prob = self.D(x_bar.detach(), hint_m)
            d_loss = self.criterion_D(batch_m, d_prob)
            
            d_loss.backward()
            self.opt_D.step()
            
            # ==========================
            # 3. 생성자(G) 학습
            # ==========================
            self.opt_G.zero_grad()
            
            x_hat = self.G(x_tilde, batch_m)
            x_bar = batch_m * batch_x + (1 - batch_m) * x_hat
            
            d_prob = self.D(x_bar, hint_m)
            g_loss, adv_loss, mse_loss = self.criterion_G(batch_x, batch_m, x_hat, d_prob)
            
            g_loss.backward()
            self.opt_G.step()
            
            # 로그 기록
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_mse_loss += mse_loss.item()
            
        return epoch_d_loss / len(train_loader), epoch_g_loss / len(train_loader), epoch_adv_loss / len(train_loader), epoch_mse_loss / len(train_loader) 

    def val_epoch(self, val_loader):
        self.G.eval() 
        
        epoch_rmse = 0.0
        
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch_x, batch_m_val, batch_m_target in pbar:
                batch_x = batch_x.to(self.device)
                batch_m_val = batch_m_val.to(self.device)
                batch_m_target = batch_m_target.to(self.device)
                
                z = torch.empty_like(batch_x).uniform_(0, 0.01).to(self.device)
                x_tilde = batch_m_val * batch_x + (1 - batch_m_val) * z
                x_hat = self.G(x_tilde, batch_m_val)
                
                # RMSE 계산
                sq_error = (batch_m_target * batch_x - batch_m_target * x_hat) ** 2
                target_count = torch.sum(batch_m_target) + 1e-8
                
                mse = torch.sum(sq_error) / target_count
                rmse = torch.sqrt(mse)
                epoch_rmse += rmse.item()
                
        return epoch_rmse / len(val_loader)