import numpy as np
from modelFKdeneme_verified import compute_fk_ur10e

def improved_numeric_ik_solver(T_target, params, initial_guess=None, max_iterations=200, tolerance=1e-6):
    """
    İyileştirilmiş nümerik IK çözücü - Daha kararlı algoritma
    """
    q = np.zeros(6) if initial_guess is None else np.array(initial_guess).copy()
    
    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]
    
    # Başlangıç sönümleme parametresi
    lambda_damping = 1e-3
    lambda_max = 1e3
    
    # Ağırlıklandırma faktörleri
    w_pos = 10.0  # Pozisyon hatasına daha fazla ağırlık
    w_ori = 1.0
    
    # Adım boyutu kontrolü
    max_step_size = 0.5  # Maksimum adım boyutu (radyan)
    
    prev_error_norm = float('inf')
    stagnation_counter = 0
    
    for iteration in range(max_iterations):
        # Mevcut FK hesapla
        fk_result = compute_fk_ur10e(q, params)
        current_pos = fk_result['T'][:3, 3]
        current_rot = fk_result['T'][:3, :3]
        
        # Pozisyon hatası
        pos_error = target_pos - current_pos
        
        # İyileştirilmiş oryantasyon hatası hesabı
        R_error = target_rot @ current_rot.T
        
        # Axis-angle representation (daha kararlı)
        # Rodriguez formula kullanarak
        trace_R = np.trace(R_error)
        if trace_R >= 3.0 - 1e-6:  # Identity matrix yakınında
            orient_error = np.zeros(3)
        else:
            # Skew-symmetric part'tan axis çıkar
            skew = (R_error - R_error.T) / 2
            orient_error = np.array([skew[2,1], skew[0,2], skew[1,0]])
            
            # Açı büyüklüğünü kontrol et
            angle = np.arccos(np.clip((trace_R - 1) / 2, -1.0, 1.0))
            if angle > 1e-6:
                orient_error = orient_error * angle / np.linalg.norm(orient_error)
        
        # Toplam hata vektörü
        error_vector = np.concatenate([w_pos * pos_error, w_ori * orient_error])
        current_error_norm = np.linalg.norm(error_vector)
        
        # Başarı kontrolü
        pos_error_norm = np.linalg.norm(pos_error)
        ori_error_norm = np.linalg.norm(orient_error)
        
        if pos_error_norm < tolerance and ori_error_norm < tolerance:
            return {
                'success': True, 
                'joint_angles': q, 
                'position_error': pos_error_norm,
                'orientation_error': ori_error_norm, 
                'iterations': iteration + 1
            }
        
        # Jakobyan hesapla
        J = compute_improved_numerical_jacobian(q, params)
        J_weighted = np.vstack([w_pos * J[:3, :], w_ori * J[3:, :]])
        
        # Damped Least Squares ile adım hesapla
        JTJ = J_weighted.T @ J_weighted
        A = JTJ + lambda_damping * np.eye(6)
        g = J_weighted.T @ error_vector
        
        try:
            delta_q = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            # Singular matrix durumunda SVD kullan
            U, s, Vt = np.linalg.svd(A)
            s_inv = np.where(s > 1e-12, 1/s, 0)
            delta_q = Vt.T @ np.diag(s_inv) @ U.T @ g
        
        # Adım boyutunu sınırla
        step_norm = np.linalg.norm(delta_q)
        if step_norm > max_step_size:
            delta_q = delta_q * max_step_size / step_norm
        
        # Yeni eklem açıları
        q_new = q + delta_q
        
        # Yeni pozisyonun hatasını kontrol et
        fk_new = compute_fk_ur10e(q_new, params)
        new_pos_error = np.linalg.norm(target_pos - fk_new['T'][:3, 3])
        
        # Armijo line search benzeri kabul kriteri
        if new_pos_error < pos_error_norm + 1e-4:
            # İlerleme var, adımı kabul et
            q = q_new
            lambda_damping = max(lambda_damping * 0.7, 1e-6)
            stagnation_counter = 0
        else:
            # İlerleme yok, sönümlemeyi artır
            lambda_damping = min(lambda_damping * 3.0, lambda_max)
            stagnation_counter += 1
        
        # Takılma durumunu kontrol et
        if abs(current_error_norm - prev_error_norm) < 1e-8:
            stagnation_counter += 1
        
        if stagnation_counter > 10:
            # Takıldı, restart yap
            q += np.random.normal(0, 0.1, 6)  # Küçük rastgele bozma
            lambda_damping = 1e-3
            stagnation_counter = 0
        
        prev_error_norm = current_error_norm
        
        # Çok büyük sönümleme = başarısızlık
        if lambda_damping >= lambda_max:
            break
    
    return {
        'success': False, 
        'joint_angles': q, 
        'position_error': np.linalg.norm(pos_error),
        'orientation_error': np.linalg.norm(orient_error), 
        'iterations': max_iterations
    }


def compute_improved_numerical_jacobian(q, params, epsilon=1e-7):
    """
    İyileştirilmiş nümerik Jakobyan hesabı
    """
    J = np.zeros((6, 6))
    
    # Referans FK
    fk_ref = compute_fk_ur10e(q, params)
    ref_pos = fk_ref['T'][:3, 3]
    ref_rot = fk_ref['T'][:3, :3]
    
    for i in range(6):
        # Forward difference
        q_plus = q.copy()
        q_plus[i] += epsilon
        fk_plus = compute_fk_ur10e(q_plus, params)
        
        # Pozisyon Jakobyası
        J[:3, i] = (fk_plus['T'][:3, 3] - ref_pos) / epsilon
        
        # Oryantasyon Jakobyası
        R_dot = (fk_plus['T'][:3, :3] - ref_rot) / epsilon
        omega_skew = ref_rot.T @ R_dot
        
        # Skew-symmetric matrisinden açısal hız vektörüne
        J[3, i] = omega_skew[2, 1]
        J[4, i] = omega_skew[0, 2]
        J[5, i] = omega_skew[1, 0]
    
    return J


def multi_start_improved_ik_solver(T_target, params, num_starts=20, **kwargs):
    """
    Çoklu başlangıç noktası ile IK çözme
    """
    solutions = []
    
    # Akıllı başlangıç noktaları
    target_pos = T_target[:3, 3]
    base_angle = np.arctan2(target_pos[1], target_pos[0])
    
    smart_starts = [
        np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0]),  # Home position
        np.array([base_angle, -np.pi/3, np.pi/3, 0, np.pi/2, 0]),  # Target oriented
        np.array([base_angle, -2*np.pi/3, 2*np.pi/3, 0, np.pi/2, 0]),  # Elbow down
        np.array([base_angle + np.pi, -np.pi/3, np.pi/3, np.pi, -np.pi/2, 0]),  # Flip config
    ]
    
    # Önce akıllı başlangıçları dene
    for start in smart_starts:
        result = improved_numeric_ik_solver(T_target, params, initial_guess=start, **kwargs)
        if result['success']:
            solutions.append(result)
            if len(solutions) >= 3:  # Yeterli çözüm bulundu
                break
    
    # Rastgele başlangıçlar
    attempts = 0
    while len(solutions) == 0 and attempts < num_starts:
        random_start = np.random.uniform(-np.pi, np.pi, 6)
        result = improved_numeric_ik_solver(T_target, params, initial_guess=random_start, **kwargs)
        if result['success']:
            solutions.append(result)
        attempts += 1
    
    return solutions if solutions else [{'success': False, 'joint_angles': np.zeros(6)}]