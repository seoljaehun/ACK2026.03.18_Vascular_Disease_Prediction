import os
import torch
from torch.utils.data import DataLoader

from utils.Config import Config
from dataset.Dataset import PreprocessedKNHANESDataset
from model.Model import Generator, Discriminator
from loss.Loss import GeneratorLoss, DiscriminatorLoss
from train.Train import GAINTrainer

torch.cuda.init()

def main():
    
    # 체크포인트 저장 폴더 생성
    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)
        print(f"모델 저장 폴더: {Config.checkpoint_dir}")

    print(f"사용 디바이스: {Config.device}")

    # =====================================================================
    # 2. GAIN 전용 파라미터 설정
    # =====================================================================
    ALPHA = 100.0            # [Loss] 원본 데이터 보존 가중치
    HINT_RATE = 0.9         # [Trainer] 판별자에게 정답을 알려줄 힌트 비율
    VAL_SPLIT_RATIO = 0.2   # [Dataset] 전체 데이터 중 검증용 데이터 분할 비율
    VAL_MASK_RATE = 0.2     # [Dataset] 검증 평가 시 인위적으로 뚫어버릴 빈칸 비율
    RANDOM_SEED = 42        # [Dataset] Train/Val 셔플링 시퀀스 고정용 시드

    # =====================================================================
    # 3. 데이터셋 및 데이터로더 초기화
    # =====================================================================
    train_dataset = PreprocessedKNHANESDataset(
        csv_path=Config.root_dir, 
        mode='train', 
        val_split_ratio=VAL_SPLIT_RATIO, 
        val_mask_rate=VAL_MASK_RATE, 
        random_seed=RANDOM_SEED
    )
    
    val_dataset = PreprocessedKNHANESDataset(
        csv_path=Config.root_dir, 
        mode='val', 
        val_split_ratio=VAL_SPLIT_RATIO, 
        val_mask_rate=VAL_MASK_RATE, 
        random_seed=RANDOM_SEED
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers
    )

    # 데이터의 변수 개수(dim) 자동 인식
    dim = train_dataset.X.shape[1]
    
    print(f"총 변수 개수 (Feature Dimension): {dim}개")
    print(f"훈련(Train) 데이터: {len(train_dataset)} 명")
    print(f"검증(Validation) 데이터: {len(val_dataset)} 명")

    # =====================================================================
    # 4. 모델, 손실 함수, Trainer 초기화
    # =====================================================================
    G = Generator(dim)
    D = Discriminator(dim)
    
    criterion_G = GeneratorLoss(alpha=ALPHA)
    criterion_D = DiscriminatorLoss()

    trainer = GAINTrainer(
        generator=G,
        discriminator=D,
        criterion_G=criterion_G,
        criterion_D=criterion_D,
        hint_rate=HINT_RATE
    )

    # =====================================================================
    # 5. 학습 루프 (Training Loop)
    # =====================================================================
    best_rmse = float('inf')

    for epoch in range(1, Config.epochs + 1):
        # 1. Train
        d_loss, g_loss, adv_loss, mse_loss = trainer.train_epoch(train_loader)
        
        # 2. Validation
        val_rmse = trainer.val_epoch(val_loader)
        
        # 3. 진행 상황 출력
        print(f"[Epoch {epoch:3d}/{Config.epochs}] D_Loss: {d_loss:.4f} | G_Loss: {g_loss:.4f} (Adv: {adv_loss:.4f}, MSE: {mse_loss:.4f}) | Val RMSE: {val_rmse:.4f}")

        # 4. 최고 성능 모델 저장 (Best Checkpoint)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_path = os.path.join(Config.checkpoint_dir, 'best_gain_model.pth')
            torch.save(G.state_dict(), save_path)
            print(f"Best Model Saved!")

    print(f"최종 베스트 RMSE 점수: {best_rmse:.4f}")
    print(f"저장된 모델 경로: {os.path.join(Config.checkpoint_dir, 'best_gain_model.pth')}")

if __name__ == '__main__':
    main()