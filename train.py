import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    data = sio.loadmat('InF_DH_FR1.mat', squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p = np.asarray(data['p'], dtype=float)
    
    num_anchors = d_hat.shape[0]
    num_users = d_hat.shape[1]
    
    X_all = []
    y_all = []
    error_threshold = 2.5 
    
    # 2. 데이터 전처리 및 라벨링 
    for u in range(num_users):
        for a in range(num_anchors):
            meas_dist = d_hat[a, u]
            
            if meas_dist <= 0 or meas_dist > 136:
                continue
                
            true_dist = np.sqrt((BS_positions[0, a] - p[0, u])**2 + (BS_positions[1, a] - p[1, u])**2)
            error = abs(meas_dist - true_dist)
            
            label = 1 if error <= error_threshold else 0
            
            BS_x = BS_positions[0, a]
            BS_y = BS_positions[1, a]
            decimal_part = meas_dist - int(meas_dist)
            
            X_all.append([meas_dist, BS_x, BS_y, decimal_part]) 
            y_all.append(label)
            
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # 3. 모델 정의 및 전체 데이터 최종 학습 
    model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    model.fit(X_all, y_all)
    
    # 4. 모델 파일 저장
    joblib.dump(model, 'adaboost_model.pkl') 

if __name__ == "__main__":
    main()