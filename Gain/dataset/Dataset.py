import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedKNHANESDataset(Dataset):
    def __init__(self, csv_path, mode='train', val_split_ratio=0.2, val_mask_rate=0.2, random_seed=42):
        """
            csv_path: 전처리가 완료된 데이터셋 CSV 파일 경로
            mode: 'train' 또는 'val'
            val_split_ratio: 전체 데이터 중 Validation으로 사용할 비율 (기본 20%)
            val_mask_rate: Validation 시 인위적으로 뚫을 결측치 비율 (기본 20%)
            random_seed: Train과 Val 데이터를 나눌 때 동일하게 섞기 위한 시드 값
        """
        self.mode = mode
        
        # 1. 데이터 로드 및 1열(ID) 제거
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.iloc[:, 1:] 
        
        # 데이터 무작위 셔플(시드 고정)
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - val_split_ratio))
        
        if self.mode == 'train':
            # 처음부터 80% 지점까지 슬라이싱
            df = df.iloc[:split_idx]
        elif self.mode == 'val':
            # 80% 지점부터 끝까지 슬라이싱
            df = df.iloc[split_idx:]
        else:
            raise ValueError("mode는 'train' 또는 'val' 이어야 합니다.")
        
        # 2. 원래 결측치 위치를 파악하는 마스크(M) 생성
        # 관측된 값은 1.0, 비어있는 값(NaN)은 0.0
        self.M = ~df.isna().values
        self.M = self.M.astype(np.float32)
        
        # 3. 데이터 배열 생성 (NaN은 딥러닝 연산을 위해 0.0으로 임시 대체)
        self.X = df.fillna(0.0).values
        self.X = self.X.astype(np.float32)

        # ==========================================
        # 4. 인위적 마스킹 (Validation 용)
        # ==========================================
        if self.mode == 'val':
            self.M_val = np.copy(self.M)
            self.M_target = np.zeros_like(self.M)

            # 원래 값이 존재하는 위치(M==1.0) 중에서
            # val_mask_rate(예: 20%) 확률로 인위적인 결측치 마스크 생성
            mask_to_hide = (np.random.rand(*self.M.shape) < val_mask_rate) & (self.M == 1.0)
            
            # 모델에게 줄 입력용 마스크에서는 해당 위치를 0.0(결측치)으로 가림
            self.M_val[mask_to_hide] = 0.0      
            
            # 나중에 정답을 채점할 타겟 마스크에서는 해당 위치만 1.0으로 표시
            self.M_target[mask_to_hide] = 1.0   

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        
        if self.mode == 'train':
            # 학습 모드: 입력 데이터와 원본 마스크 반환
            
            m = torch.tensor(self.M[idx])
            
            return x, m
            
        elif self.mode == 'val':
            # 검증 모드: 입력 데이터, 가려진 마스크, 채점용 타겟 마스크 반환
            
            m_val = torch.tensor(self.M_val[idx])
            m_target = torch.tensor(self.M_target[idx])
            
            return x, m_val, m_target