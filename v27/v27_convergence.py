
import json, numpy as np, matplotlib.pyplot as plt

def nb_bd_kernel(N: int) -> np.ndarray:
    m = np.arange(1, N+1).reshape(-1,1)
    n = np.arange(1, N+1).reshape(1,-1)
    return np.exp(-0.5*np.abs(np.log(m/n)))

def fro_norm(K: np.ndarray) -> float:
    return float(np.linalg.norm(K, 'fro'))

def sym_eigs(G: np.ndarray) -> np.ndarray:
    Gs = (G + G.T)/2.0
    return np.sort(np.linalg.eigvalsh(Gs))

def entropy_from_eigs(w: np.ndarray) -> float:
    w = np.clip(w, 0.0, None); s = float(w.sum())
    if s<=0: return 0.0
    p = w/s; p = p[p>0]
    return float(-(p*np.log(p)).sum())

def finite_diff_dH_dbeta(K: np.ndarray, beta: float, h: float=1e-3) -> float:
    F = fro_norm(K)
    def H_of(b):
        A = K/(F**b); w = sym_eigs(A); return entropy_from_eigs(w)
    return (H_of(beta+h) - H_of(beta-h))/(2*h)

def simulate(N=200, steps=60, beta0=0.5, gamma=0.2, eps0=0.02, eps_decay=0.98, k_top=16, seed=0):
    rng = np.random.default_rng(seed)
    K0 = nb_bd_kernel(N); F = fro_norm(K0)
    beta = beta0; eps = eps0
    logs = {"t": [], "beta": [], "entropy": [], "lambda_min": [], "R": []}
    eig_traj = []

    A0 = K0/(F**beta); w0 = sym_eigs(A0); H0 = entropy_from_eigs(w0); lmin0 = float(w0.min())

    for t in range(steps+1):
        noise = rng.normal(0.0, 1.0, size=K0.shape); noise = noise / max(1.0, fro_norm(noise))
        Kt = (K0/(F**beta)) + eps*noise
        w = sym_eigs(Kt); H = entropy_from_eigs(w); lmin = float(w.min())
        top = w[-k_top:] if w.size>=k_top else w; eig_traj.append(top.astype(float))

        R = (H-H0) + (np.log(lmin/lmin0) if (lmin>0 and lmin0>0) else -1e9)
        logs["t"].append(t); logs["beta"].append(float(beta)); logs["entropy"].append(float(H)); logs["lambda_min"].append(lmin); logs["R"].append(float(R))
        if t==steps: break
        # update
        dHdb = finite_diff_dH_dbeta(K0, beta)
        beta = float(np.clip(beta - gamma*dHdb, 0.0, 1.0)); eps = float(eps*eps_decay)

    # Save results
    res = {"meta":{"N":N,"steps":steps,"beta0":beta0,"gamma":gamma,"eps0":eps0,"eps_decay":eps_decay,"k_top":k_top,"seed":seed},
           "logs": logs, "eig_traj": np.array(eig_traj, dtype=float).tolist()}
    with open("results_v27.json","w",encoding="utf-8") as f: json.dump(res,f,ensure_ascii=False,indent=2)

    # Plots
    plt.figure(); plt.plot(logs["t"], logs["entropy"]); plt.title("Entropy vs time (v27)"); plt.xlabel("time"); plt.ylabel("entropy")
    plt.savefig("figs/entropy_flow.png", dpi=160); plt.close()

    plt.figure(); plt.plot(logs["t"], logs["beta"]); plt.title("Beta evolution (v27)"); plt.xlabel("time"); plt.ylabel("beta_t")
    plt.savefig("figs/beta_evolution.png", dpi=160); plt.close()

    E = np.array(eig_traj, dtype=float)
    plt.figure(); plt.imshow(E, aspect="auto"); plt.title("Top-k eigenvalues over time"); plt.xlabel("rank (top-k)"); plt.ylabel("time step"); plt.colorbar()
    plt.savefig("figs/eigen_heatmap.png", dpi=160); plt.close()

    if E.size>0:
        last = E[-1]; x = np.arange(1, len(last)+1)
        plt.figure(); plt.scatter(x, last); plt.title("Critical alignment (final snapshot)"); plt.xlabel("rank"); plt.ylabel("lambda")
        plt.savefig("figs/critical_alignment.png", dpi=160); plt.close()

    print("[v27] Done. See results_v27.json and figs/*.png")

if __name__ == "__main__":
    simulate()
