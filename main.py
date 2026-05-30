import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
import joblib
import os

# 1. HDOP 계산 함수
def calc_hdop(anchors, est_pos):
    A = []
    for i in range(anchors.shape[1]):
        dx = est_pos[0] - anchors[0, i]
        dy = est_pos[1] - anchors[1, i]
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0: continue
        A.append([dx/dist, dy/dist])
    A = np.array(A)
    try:
        H = np.linalg.inv(A.T @ A)
        return np.sqrt(H[0, 0] + H[1, 1])
    except np.linalg.LinAlgError:
        return float('inf')

# 2. WLS를 위한 잔차 함수
def residuals(pos, anchors, distances, weights):
    res = []
    for i in range(anchors.shape[1]):
        dist_calc = np.sqrt((pos[0] - anchors[0, i])**2 + (pos[1] - anchors[1, i])**2)
        res.append(weights[i] * (dist_calc - distances[i]))
    return res

# 3. 메인 측위 알고리즘 
def your_algorithm(d_hat_u, p_bs, ml_model):
    valid_mask = (d_hat_u > 0) & (d_hat_u <= 136)
    valid_distances = d_hat_u[valid_mask]
    valid_anchors = p_bs[:, valid_mask]
    num_valid = len(valid_distances)
    
    if num_valid < 3:
        return np.array([0.0, 0.0])

    if ml_model is not None:
        valid_BS_x = valid_anchors[0, :]
        valid_BS_y = valid_anchors[1, :]
        decimal_parts = valid_distances - valid_distances.astype(int)
        
        X_test = np.column_stack((valid_distances, valid_BS_x, valid_BS_y, decimal_parts))
        quality_scores = ml_model.predict_proba(X_test)[:, 1] 
    else:
        quality_scores = np.ones(num_valid)


    k_best = min(7, num_valid) 
    top_indices = np.argsort(quality_scores)[-k_best:]
    
    valid_anchors = valid_anchors[:, top_indices]
    valid_distances = valid_distances[top_indices]
    quality_scores = quality_scores[top_indices]
    
    num_valid = len(valid_distances)
    if num_valid < 3:
        return np.array([0.0, 0.0])

    best_pos = np.mean(valid_anchors, axis=1)
    min_cost = float('inf') 
    

    max_iter = 80 
    anchor_indices = np.arange(num_valid)
    
    sum_scores = np.sum(quality_scores)
    if sum_scores == 0:
        prob_dist = np.ones(num_valid) / num_valid
    else:
        prob_dist = quality_scores / sum_scores

    for _ in range(max_iter):
    
        sample_size = min(3, num_valid) 
        sample_idx = np.random.choice(anchor_indices, sample_size, replace=False, p=prob_dist)
        
        sample_anchors = valid_anchors[:, sample_idx]
        sample_distances = valid_distances[sample_idx]
        sample_weights = quality_scores[sample_idx]
        
        initial_guess = np.mean(sample_anchors, axis=1)

  
        result = least_squares(residuals, initial_guess, args=(sample_anchors, sample_distances, sample_weights), loss='linear')
        est_pos = result.x
        
        est_pos[0] = np.clip(est_pos[0], -70.0, 70.0)
        est_pos[1] = np.clip(est_pos[1], -70.0, 70.0)
        
        # HDOP 기하학 연산
        current_hdop = calc_hdop(sample_anchors, est_pos)
        
        # 전체 앵커 간의 잔차 합 계산
        all_calc_dists = np.sqrt(np.sum((valid_anchors.T - est_pos)**2, axis=1))
        all_errors = np.abs(all_calc_dists - valid_distances)
        current_cost = np.sum(quality_scores * all_errors)
        
        safe_hdop = min(current_hdop, 10.0)
        Final_score = current_cost * (safe_hdop ** 1.5)
        
        if Final_score < min_cost:
            min_cost = Final_score
            best_pos = est_pos
            
    return best_pos


def main():
    # 1) 머신러닝 가중치 모델 및 데이터 로드
    model_path = 'model.pkl'
    ml_model = None
    if os.path.exists(model_path):
        ml_model = joblib.load(model_path)

    mat_path = 'DH_FR1.mat' 
    try:
        data = sio.loadmat(mat_path, squeeze_me=False)
    except FileNotFoundError:
        try:
            data = sio.loadmat('InF_DH_FR1.mat', squeeze_me=False)
        except FileNotFoundError:
            return

    BS_positions = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    # 2)  사용자 수: 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions, ml_model)


    # 1. CSV FILE
    np.savetxt("estimated_positions.csv", p_hat.T, delimiter=",", header="X,Y", comments="")

    # 2. MAT FILE
    sio.savemat('estimated_positions.mat', {'p_hat': p_hat})



    # 3) 결과 반환
    return p_hat

if __name__ == "__main__":
    main()