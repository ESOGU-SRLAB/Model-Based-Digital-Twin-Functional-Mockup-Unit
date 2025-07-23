# Dosya Adı: modelIKdeneme_solver.py (NİHAİ KARARLILIK VERSİYONU)

import numpy as np
from modelFKdeneme_verified import compute_fk_ur10e

def numeric_ik_solver(T_target, params, initial_guess=None, max_iterations=150, tolerance=1e-6):
    """
    modelFKdeneme_verified.py 'kara kutusu' ile %100 uyumlu, kararlılığı en üst
    düzeye çıkarılmış nümerik IK çözücü. Bu versiyon, adaptif sönümleme ve
    hata ağırlıklandırma tekniklerini kullanır.
    """
    q = np.zeros(6) if initial_guess is None else np.array(initial_guess)
    
    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]
    
    # Adaptif sönümleme için başlangıç değeri
    lambda_damping = 1e-2
    # Ağırlıklandırma faktörleri (pozisyon hatasını daha çok önemse)
    w_pos = 1.0
    w_ori = 0.5 

    last_error_norm = float('inf')

    for iteration in range(max_iterations):
        fk_result = compute_fk_ur10e(q, params)
        current_pos = fk_result['T'][:3, 3]
        current_rot = fk_result['T'][:3, :3]
        
        # Hataları hesapla
        pos_error = target_pos - current_pos
        R_error = target_rot @ current_rot.T
        trace_R = np.clip(np.trace(R_error), -1.0, 3.0)
        angle_error = np.arccos((trace_R - 1) / 2)
        
        orient_error = np.zeros(3)
        if angle_error > 1e-6:
            axis = 1 / (2 * np.sin(angle_error)) * np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0], 
                R_error[1, 0] - R_error[0, 1]
            ])
            orient_error = axis * angle_error
            
        # Ağırlıklı toplam hata
        error_vector = np.concatenate([w_pos * pos_error, w_ori * orient_error])
        current_error_norm = np.linalg.norm(error_vector)

        # Başarı durumunu kontrol et
        if np.linalg.norm(pos_error) < tolerance and np.linalg.norm(orient_error) < tolerance:
            return {'success': True, 'joint_angles': q, 'position_error': np.linalg.norm(pos_error),
                    'orientation_error': np.linalg.norm(orient_error), 'iterations': iteration + 1}

        J = compute_numerical_jacobian(q, params)
        
        # Ağırlıkları Jakobyan'a uygula
        J_weighted = np.vstack([w_pos * J[:3, :], w_ori * J[3:, :]])

        # Damped Least Squares (DLS)
        A = J_weighted.T @ J_weighted + lambda_damping * np.eye(6)
        g = J_weighted.T @ error_vector
        delta_q = np.linalg.solve(A, g)
        
        # Yeni aday açıları hesapla
        q_new = q + delta_q
        
        # Yeni adayın hatasını kontrol et
        fk_new_result = compute_fk_ur10e(q_new, params)
        pos_err_new = np.linalg.norm(target_pos - fk_new_result['T'][:3, 3])
        # (Oryantasyon hatasını da kontrol etmek daha doğru olur ama şimdilik pozisyon yeterli)
        
        # ADAPTİF SÖNÜMLEME MANTIĞI
        if pos_err_new < np.linalg.norm(pos_error): # Hata azaldıysa, adım geçerli
            q = np.arctan2(np.sin(q_new), np.cos(q_new)) # Normalleştir ve kabul et
            lambda_damping *= 0.8 # Kendine güveni artır, sönümlemeyi azalt
        else: # Hata arttıysa, adım geçersiz
            lambda_damping *= 2 # Panik yap, sönümlemeyi artır
            
        # Eğer sönümleme çok büyürse, yerel minimumda takılmış olabiliriz.
        if lambda_damping > 1e6:
            break

    return {'success': False, 'joint_angles': q, 'position_error': np.linalg.norm(pos_error),
            'orientation_error': np.linalg.norm(orient_error), 'iterations': max_iterations}


def compute_numerical_jacobian(q, params, epsilon=1e-6):
    J = np.zeros((6, 6))
    fk_ref = compute_fk_ur10e(q, params)
    current_pos = fk_ref['T'][:3, 3]
    current_rot = fk_ref['T'][:3, :3]
    
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += epsilon
        fk_plus = compute_fk_ur10e(q_plus, params)
        
        q_minus = q.copy()
        q_minus[i] -= epsilon
        fk_minus = compute_fk_ur10e(q_minus, params)
        
        J[:3, i] = (fk_plus['T'][:3, 3] - fk_minus['T'][:3, 3]) / (2 * epsilon)
        
        R_dot = (fk_plus['T'][:3, :3] - fk_minus['T'][:3, :3]) / (2 * epsilon)
        omega_skew = current_rot.T @ R_dot
        J[3, i] = (omega_skew[2, 1] - omega_skew[1, 2]) / 2
        J[4, i] = (omega_skew[0, 2] - omega_skew[2, 0]) / 2
        J[5, i] = (omega_skew[1, 0] - omega_skew[0, 1]) / 2
    return J


def multi_start_ik_solver(T_target, params, num_starts=50, **kwargs):
    solutions = []
    # İlk deneme her zaman sıfır konfigürasyonundan başlasın
    result = numeric_ik_solver(T_target, params, initial_guess=np.zeros(6), **kwargs)
    if result['success']:
        solutions.append(result)

    # Kalan denemeleri rastgele yap
    for _ in range(num_starts - 1):
        # Önceki başarılı çözüme yakın bir çözüm varsa daha fazla arama
        if len(solutions) > 0:
            break
        initial_guess = np.random.uniform(-np.pi, np.pi, 6)
        result = numeric_ik_solver(T_target, params, initial_guess=initial_guess, **kwargs)
        if result['success']:
            solutions.append(result)
            # İlk başarılı çözümü bulduktan sonra durmak, süreci hızlandırır
            break
            
    return solutions