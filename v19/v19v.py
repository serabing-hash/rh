import numpy as np
import scipy.linalg as la
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 1) 커널 생성: K_{mn} = sqrt(min(m,n)/max(m,n))
def build_K(N: int) -> np.ndarray:
    idx = np.arange(1, N+1, dtype=float)
    M = idx[:, None]
    Nn = idx[None, :]
    K = np.sqrt(np.minimum(M, Nn) / np.maximum(M, Nn))
    return K

# 2) Clark-프록시 측도 (e1에 대한 스펙트럼 질량)
def spectral_measure_e1(K: np.ndarray):
    # 고유분해 (대칭행렬이므로 eigh)
    evals, evecs = la.eigh(K, overwrite_a=True, check_finite=False)
    # 가중치: (v_i · e1)^2 -> 첫 성분 제곱
    w = evecs[0, :]**2
    return evals, w

# 3) 지표 계산
def atomness_metrics(evals, w, kde_bw='scott'):
    # (a) top-weight ratio
    med_w = np.median(w)
    top_weight_ratio = (w.max() / med_w) if med_w > 0 else np.inf

    # (b) KDE로 밀도 첨두비
    kde = gaussian_kde(evals, bw_method=kde_bw, weights=None)  # 고유값 분포 자체
    grid = np.linspace(evals.min(), evals.max(), 512)
    dens = kde(grid)
    med_d = np.median(dens)
    peak_ratio = (dens.max() / med_d) if med_d > 0 else np.inf

    # (c) 간격지표
    gaps = np.diff(np.sort(evals))
    gaps = gaps[gaps > 1e-14]  # 수치적 0 간격 제거
    med_g = np.median(gaps) if gaps.size else np.nan
    gap_ratio = (gaps.max() / med_g) if (gaps.size and med_g > 0) else np.nan

    return {
        'top_weight_ratio': float(top_weight_ratio),
        'peak_ratio': float(peak_ratio),
        'gap_ratio': float(gap_ratio),
        'grid': grid,
        'density': dens
    }

# 4) 실행 & 도식
def run_once(N=600, kde_bw='scott', show=True, save_prefix=None):
    K = build_K(N)
    evals, w = spectral_measure_e1(K)
    met = atomness_metrics(evals, w, kde_bw=kde_bw)

    if show or save_prefix:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        # (i) Clark-프록시 질량 w_i 분포
        ax[0].plot(np.sort(w)[::-1], marker='.', lw=1)
        ax[0].set_title(f'w_i sorted (N={N})\nTop/Median={met["top_weight_ratio"]:.2f}')
        ax[0].set_xlabel('rank'); ax[0].set_ylabel('weight')

        # (ii) 고유값 분포 + KDE
        ax[1].hist(evals, bins=40, density=True, alpha=0.35, edgecolor='k')
        ax[1].plot(met['grid'], met['density'], lw=2)
        ax[1].set_title(f'λ-density (KDE peak/median={met["peak_ratio"]:.2f})')
        ax[1].set_xlabel('eigenvalue'); ax[1].set_ylabel('density')

        # (iii) 간격 히스토그램
        gaps = np.diff(np.sort(evals))
        gaps = gaps[gaps > 1e-14]
        ax[2].hist(gaps, bins=40, density=True, alpha=0.6, edgecolor='k')
        ax[2].set_title(f'gap ratio (max/median) = {met["gap_ratio"]:.2f}')
        ax[2].set_xlabel('gap'); ax[2].set_ylabel('density')

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f'{save_prefix}_summary.png', dpi=180)
        if show:
            plt.show()

    return evals, w, met

# 5) 해상도 검사: 여러 N에서 지표 비교
def resolution_sweep(N_list=(300, 600, 900, 1200), kde_bw='scott'):
    rows = []
    for N in N_list:
        evals, w, met = run_once(N=N, kde_bw=kde_bw, show=False, save_prefix=None)
        rows.append((N, met['top_weight_ratio'], met['peak_ratio'], met['gap_ratio']))
    return np.array(rows)

# 예시 실행:
if __name__ == "__main__":
    # 단일 실행
    evals, w, met = run_once(N=600, kde_bw='scott', show=True, save_prefix='v19D')

    # 해상도 스윕
    table = resolution_sweep(N_list=(300, 600, 900), kde_bw='scott')
    print("N, top_weight/median, KDE_peak/median, gap_max/median")
    print(table)