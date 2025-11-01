# PINN/LNN for TVP on [a,b] with hard matching at BOTH ends via quintic base
# Fast and tight: Adam (~8-10 min on Colab T4) + short L-BFGS

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize
import pandas as pd

tf.random.set_seed(7); np.random.seed(7)
DT = tf.float32

# -------------------- Problem --------------------
a_val, b_val = 0.0, 1.0
k_val, c_val, d_val = 0.5, 0.05, 0.1

# terminal (at t=b)
y_b_val, dy_b_val, ddy_b_val = 0.1, 0.2, 0.3

a = tf.constant(a_val, DT); b = tf.constant(b_val, DT)
k = tf.constant(k_val, DT); c = tf.constant(c_val, DT); d = tf.constant(d_val, DT)

# Toggle: use left-end {y, y', y''} from reference (tight match)
USE_LEFT_FROM_REF = True

# -------------------- Reference via shooting --------------------
def ode_system(t, Y):
    y1, y2, y3 = Y
    return [y2, y3, -d_val*y3 - c_val*(y2**2) - k_val*y1*np.sin(y1)]

def shoot(initial):
    sol = solve_ivp(ode_system, [a_val, b_val], initial,
                    method='LSODA', rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol.y[:, -1] - np.array([y_b_val, dy_b_val, ddy_b_val], float)

# initial guess for shooting
L = (b_val - a_val)
guess = np.array([
    y_b_val  - L*dy_b_val + 0.5*L*L*ddy_b_val,
    dy_b_val - L*ddy_b_val,
    ddy_b_val
], float)

res = root(shoot, guess, method='hybr', tol=1e-8)
if not res.success:
    raise RuntimeError("Shooting failed; try different guess.")

init = res.x
sol = solve_ivp(ode_system, [a_val, b_val], init,
                method='LSODA', t_eval=np.linspace(a_val, b_val, 801),
                rtol=1e-8, atol=1e-10, max_step=0.05)

t_ref = sol.t.astype(np.float32)
y_ref = sol.y[0].astype(np.float32)
dy_ref = sol.y[1].astype(np.float32)
ddy_ref = np.gradient(dy_ref, t_ref).astype(np.float32)

# left end (from reference)
y_a_val   = float(y_ref[0])
dy_a_val  = float(dy_ref[0])
ddy_a_val = float(ddy_ref[0])

# -------------------- Quintic base P(t) --------------------
# Build P(s) with s=(t-a)/(b-a) so that:
# P(0)=ya, P'(0)=T*dya, P''(0)=T^2*ddya; P(1)=yb, P'(1)=T*dyb, P''(1)=T^2*ddyb
def build_quintic_coeffs(ya, dya, ddya, yb, dyb, ddyb, T):
    # Solve A c = b for c = [c0..c5] of P(s)=c0+c1 s+...+c5 s^5
    A = np.array([
        [1,0,0,0,0,0],                 # P(0)=ya
        [0,1,0,0,0,0],                 # P'(0)=T*dya
        [0,0,2,0,0,0],                 # P''(0)=T^2*ddya
        [1,1,1,1,1,1],                 # P(1)=yb
        [0,1,2,3,4,5],                 # P'(1)=T*dyb
        [0,0,2,6,12,20],               # P''(1)=T^2*ddyb
    ], float)
    rhs = np.array([ya, T*dya, T*T*ddya, yb, T*dyb, T*T*ddyb], float)
    c = np.linalg.solve(A, rhs)
    return c

T = b_val - a_val
if USE_LEFT_FROM_REF:
    ya, dya, ddya = y_a_val, dy_a_val, ddy_a_val
else:
    # if you don't want to use left-end ref, anchor with zeros (still works, looser fit)
    ya, dya, ddya = 0.0, 0.0, 0.0

c_quint = build_quintic_coeffs(ya, dya, ddya, y_b_val, dy_b_val, ddy_b_val, T).astype(np.float32)

def P_base(t_tf):
    # t_tf: shape [N,1]
    s = (t_tf - a) / (b - a)
    s1 = s; s2 = s*s; s3 = s2*s; s4 = s3*s; s5 = s4*s
    c0, c1, c2, c3, c4, c5 = [tf.constant(v, DT) for v in c_quint]
    return c0 + c1*s1 + c2*s2 + c3*s3 + c4*s4 + c5*s5

def S_mask(t_tf):
    # vanish with multiplicity 3 at both ends
    s = (t_tf - a) / (b - a)
    return tf.pow(s*(1.0 - s), 3)

# -------------------- Legendre feature layer --------------------
def legendre_polys(xi, M):
    xi = tf.convert_to_tensor(xi, dtype=DT)
    L0 = tf.ones_like(xi)
    if M == 1: return L0
    L1 = xi
    outs = [L0, L1]
    for m in range(2, M):
        outs.append(((2*m-1)*xi*outs[-1] - (m-1)*outs[-2]) / m)
    return tf.concat(outs[:M], axis=-1)

# -------------------- Model --------------------
class Core(tf.keras.Model):
    def __init__(self, M=16, n_hid=32):
        super().__init__()
        self.M = M
        self.d1 = tf.keras.layers.Dense(n_hid, activation='tanh')
        self.d2 = tf.keras.layers.Dense(n_hid, activation='tanh')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, t):
        t = tf.reshape(tf.convert_to_tensor(t, DT), (-1,1))
        xi = (2.0*(t - a)/(b - a)) - 1.0
        L  = legendre_polys(xi, self.M)
        h  = self.d1(L); h = self.d2(h)
        return self.out(h)

class Model(tf.keras.Model):
    # y(t) = P(t) + S(t) * core(t)
    def __init__(self, core):
        super().__init__()
        self.core = core
    
    def call(self, t):
        t = tf.reshape(tf.convert_to_tensor(t, DT), (-1,1))
        return P_base(t) + S_mask(t) * self.core(t)

def get_derivatives(model, t):
    t = tf.reshape(tf.convert_to_tensor(t, DT), (-1,1))
    with tf.GradientTape(persistent=True) as g3:
        g3.watch(t)
        with tf.GradientTape(persistent=True) as g2:
            g2.watch(t)
            with tf.GradientTape() as g1:
                g1.watch(t)
                y = model(t)
            dy = g1.gradient(y, t)
        ddy = g2.gradient(dy, t)
    dddy = g3.gradient(ddy, t)
    del g2; del g3
    return y, dy, ddy, dddy

# -------------------- Collocation (LGL + RAR) --------------------
def lgl_nodes_weights_on_interval(N, ta, tb):
    if N < 2: raise ValueError("N>=2")
    from numpy.polynomial import legendre as npleg
    ccoef  = np.zeros(N); ccoef[-1] = 1.0
    dccoef = npleg.legder(ccoef)
    roots  = np.sort(npleg.legroots(dccoef).real)
    x  = np.concatenate(([-1.0], roots, [1.0]))  # in [-1,1]
    PNm1 = npleg.legval(x, ccoef)
    w = np.empty_like(x)
    w[0] = w[-1] = 2.0/(N*(N-1))
    w[1:-1] = 2.0/(N*(N-1)) / (PNm1[1:-1]**2)
    t = (ta + tb)/2.0 + (tb - ta)*x/2.0
    w = w * (tb - ta)/2.0
    return t.astype(np.float32).reshape(-1,1), w.astype(np.float32).reshape(-1,1)

def residual_sq(model, t_np):
    t_tf = tf.convert_to_tensor(t_np.reshape(-1,1), DT)
    y, dy, ddy, dddy = get_derivatives(model, t_tf)
    res = dddy + d*ddy + c*tf.square(dy) + k*y*tf.sin(y)
    return tf.square(res).numpy().reshape(-1)

def build_collocation(model, N_total=768):
    # half LGL, half RAR (importance)
    N_lgl = N_total // 2
    t_lgl_np, w_lgl_np = lgl_nodes_weights_on_interval(N_lgl, a_val, b_val)
    
    grid = np.linspace(a_val, b_val, 1500, dtype=np.float32)
    R2 = residual_sq(model, grid) + 1e-12
    p  = R2 / R2.sum()
    N_imp = N_total - N_lgl
    idx = np.random.choice(len(grid), size=N_imp, replace=True, p=p)
    t_imp_np = grid[idx][:,None].astype(np.float32)
    w_imp_np = np.full_like(t_imp_np, (b_val-a_val)/N_imp, dtype=np.float32)
    
    t_c_np = np.vstack([t_lgl_np, t_imp_np])
    w_c_np = np.vstack([w_lgl_np, w_imp_np])
    t_c_np = np.clip(t_c_np, a_val+1e-3, b_val-1e-3)
    return (tf.convert_to_tensor(t_c_np, DT),
            tf.convert_to_tensor(w_c_np, DT))

@tf.function
def physics_loss_weighted(model, t_c, w_c):
    y, dy, ddy, dddy = get_derivatives(model, t_c)
    res  = dddy + d*ddy + c*tf.square(dy) + k*y*tf.sin(y)
    num  = tf.reduce_sum(w_c * tf.square(res))
    den  = tf.reduce_sum(w_c)
    return num/den

@tf.function
def train_step(model, opt, t_c, w_c):
    with tf.GradientTape() as tape:
        L = physics_loss_weighted(model, t_c, w_c)
    grads = tape.gradient(L, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) if g is not None else None for g in grads]
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return L

# -------------------- Build & Train --------------------
core  = Core(M=16, n_hid=32)
model = Model(core)

N_total = 768     # 768--1024 is fine
epochs  = 900
lr0     = 3e-4
opt = tf.keras.optimizers.Adam(lr0)

loss_hist = []
prev = np.inf

for ep in range(epochs):
    t_c, w_c = build_collocation(model, N_total=N_total)
    L = train_step(model, opt, t_c, w_c)
    val = L.numpy().item()
    loss_hist.append(val)
    
    if ep % 150 == 0:
        yb, dyb, ddyb, _ = get_derivatives(model, tf.constant([[b_val]], DT))
        print(f"Epoch {ep:4d} | loss={val:.3e} | @b: y={float(yb):+.4f} "
              f"dy={float(dyb):+.4f} ddy={float(ddyb):+.4f}")

# -------------------- Short L-BFGS polish --------------------
t_c, w_c = build_collocation(model, N_total=1024)
vars_list = model.trainable_variables
shapes = [v.shape for v in vars_list]
sizes  = [int(np.prod(s)) for s in shapes]

def pack():
    return np.concatenate([v.numpy().reshape(-1) for v in vars_list]).astype(np.float64)

def unpack(x):
    off=0
    for v, sz, shp in zip(vars_list, sizes, shapes):
        v.assign(x[off:off+sz].reshape(shp).astype(np.float32))
        off+=sz

@tf.function
def full_loss():
    return physics_loss_weighted(model, t_c, w_c)

def f_and_g(x):
    unpack(x)
    with tf.GradientTape() as tape:
        L = full_loss()
    g = tape.gradient(L, vars_list)
    g_flat = np.concatenate([gi.numpy().reshape(-1) for gi in g]).astype(np.float64)
    return float(L.numpy().astype(np.float64)), g_flat

print("L-BFGS polish ...")
x0 = pack()
res_lbfgs = minimize(lambda z: f_and_g(z), x0, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=160, ftol=1e-12, gtol=1e-8))
unpack(res_lbfgs.x)
print("L-BFGS:", res_lbfgs.message)

# -------------------- Evaluate & save --------------------
t_plot = np.linspace(a_val, b_val, 801, dtype=np.float32)[:,None]
y_pinn = model(tf.convert_to_tensor(t_plot, DT)).numpy().reshape(-1)

plt.figure(figsize=(8,6))
plt.plot(t_plot, y_pinn, label='PINN')
plt.plot(t_ref, y_ref, '--', label='Reference (shooting)')
plt.xlabel('t'); plt.ylabel('y(t)'); plt.title('PINN vs Reference (tight fit)')
plt.legend(); plt.tight_layout(); plt.savefig('pinn_vs_ref.png', dpi=150); plt.show()

plt.figure(figsize=(8,6))
plt.plot(loss_hist); plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Adam loss history'); plt.tight_layout(); plt.savefig('loss_history.png', dpi=150); plt.show()

y_ref_interp = np.interp(t_plot.flatten().astype(float), t_ref, y_ref)
pd.DataFrame({'t': t_plot.flatten(),
              'y_pinn': y_pinn,
              'y_ref': y_ref_interp}).to_csv('results.csv', index=False)

print("Saved: pinn_vs_ref.png, loss_history.png, results.csv")