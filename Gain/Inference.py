import os
import torch
import pandas as pd
import numpy as np

from utils.Config import Config
from model.Model import Generator

def main():
    # =====================================================================
    # 1. 원본 데이터 로드 및 분리 (ID 보존)
    # =====================================================================
    print(f"원본 데이터 로드 중: {Config.root_dir}")
    df = pd.read_csv(Config.root_dir)
    
    # 1열(ID) 제거
    id_col_name = df.columns[0]
    ids = df.iloc[:, 0].values
    
    # 순수 피처(Feature) 데이터만 추출
    features_df = df.iloc[:, 1:]
    columns = features_df.columns
    
    # 마스크(M) 생성 (값이 있으면 1, NaN이면 0)
    mask = ~features_df.isna().values
    mask = mask.astype(np.float32)
    
    # 딥러닝 연산을 위해 NaN을 0.0으로 임시 대체한 데이터 배열 생성
    X = features_df.fillna(0.0).values
    X = X.astype(np.float32)

    # 전체 데이터를 처리하기 위해 텐서로 변환
    X_tensor = torch.tensor(X).to(Config.device)
    M_tensor = torch.tensor(mask).to(Config.device)
    
    dim = X.shape[1]
    num_samples = X.shape[0]

    # =====================================================================
    # 2. 훈련된 생성자(Generator) 모델 불러오기
    # =====================================================================
    G = Generator(dim).to(Config.device)
    
    # best_gain_model.pth 경로 설정
    model_path = os.path.join(Config.checkpoint_dir, 'best_gain_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"훈련된 모델 가중치를 찾을 수 없습니다: {model_path}")
        
    # 가중치 덮어쓰기 및 평가 모드 전환
    G.load_state_dict(torch.load(model_path, map_location=Config.device))
    G.eval()
    print(" 최고 성능의 Generator 가중치 로드 완료")

    # =====================================================================
    # 3. 추론 (Inference) 진행 - 결측치 채우기
    # =====================================================================
    print("결측치 복원 연산 진행 중...")
    
    # 최종 결과물을 담을 빈 배열 준비
    imputed_X = np.zeros_like(X)
    
    batch_size = Config.batch_size
    
    # 메모리 부족 방지를 위해 배치 단위로 나누어서 추론
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            
            batch_x = X_tensor[i:end]
            batch_m = M_tensor[i:end]
            
            # 1. 노이즈 Z 생성 및 초기 데이터(X_tilde) 세팅
            z = torch.rand_like(batch_x)
            x_tilde = batch_m * batch_x + (1 - batch_m) * z
            
            # 2. Generator 예측
            x_hat = G(x_tilde, batch_m)
            
            # 3. 최종 데이터 합성
            # 원본 값이 있던 곳(M=1)은 원래 값(batch_x)을 그대로 유지하고,
            # 비어있던 결측치(M=0) 자리만 모델이 예측한 값(x_hat)으로 덮어씌움
            batch_imputed = batch_m * batch_x + (1 - batch_m) * x_hat
            
            # Numpy 배열로 변환하여 저장
            imputed_X[i:end] = batch_imputed.cpu().numpy()

    # =====================================================================
    # 4. 결과 저장 (새로운 CSV 파일 생성)
    # =====================================================================
    # 완성된 데이터를 다시 Pandas DataFrame으로 변환
    imputed_df = pd.DataFrame(imputed_X, columns=columns)
    
    # ID 컬럼 추가
    imputed_df.insert(0, id_col_name, ids)
    
    # 원본 파일이 있는 폴더에 새 이름으로 저장
    save_dir = os.path.dirname(Config.root_dir)
    save_path = os.path.join(save_dir, "HN13_24_ALL_MERGED_Imputed_Final.csv")
    
    imputed_df.to_csv(save_path, index=False)

    print("결측치 복원이 모두 완료되었습니다!")
    print(f"저장된 경로: {save_path}")

if __name__ == '__main__':
    main()