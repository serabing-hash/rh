import numpy as np
import matplotlib.pyplot as plt

# 1) 샘플 격자/가중
M = 600                    # 총 N = 2M+1
Delta = 0.02
t = np.arange(-M, M+1) * Delta
N = t.size
alpha = 5e-3               # 테이퍼 강도
w = np.exp(-alpha * t**2)

# 2) 커널 행렬 K_N
# K(s,t) = 1/cosh(pi*(s-t))
# 벡터화 계산
T1 = t[:,None]
T2 = t[None,:]
K = 1.0/np.cosh(np.pi*(T1 - T2))

# 가중 포함 (좌우 대칭 테이퍼)
W = np.diag(w)
K_N = W @ K @ W

# 3) 고유분해
vals, vecs = np.linalg.eigh(K_N)   # symmetric PSD
vals = np.maximum(vals, 0)         # 수치 음수 제거

# 4) 지표 시각화
plt.figure()
plt.hist(vals, bins=80, density=True)
plt.title("Eigenvalue density (approx. spectral measure)")
plt.xlabel("lambda"); plt.ylabel("density")
plt.tight_layout(); plt.show()

# 누적 분포
sv = np.sort(vals)
F = np.arange(1, N+1)/N
plt.figure()
plt.plot(sv, F, lw=2)
plt.title("Cumulative spectral function F_N")
plt.xlabel("lambda"); plt.ylabel("F_N(lambda)")
plt.tight_layout(); plt.show()

# 5) 민감도 테스트: alpha, Delta, M 바꿔 반복