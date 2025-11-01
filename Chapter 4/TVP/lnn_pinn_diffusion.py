# ==== LNN-PINN for 2nd-order TVP (Legendre features + composite loss on LGL nodes) ====
# Error-free: separates results CSV (grid-sized) and training-history CSV (epoch-sized)

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legder, legroots
from scipy.integrate import solve_ivp
import pandas as pd

# -------------------- precision & seeds --------------------
tf.keras.backend.set_floatx('float64')
np.random.seed(42); tf.random.set_seed(42)

# -------------------- domain --------------------
a, b = 0.0, 5.0   # [a,b] with terminal at t=b

# -------------------- 2nd-order RHS: y'' = f(t, y, y') --------------------
# Example TVP (damped pendulum): y'' + beta*y' + sin(y) = 0  ->  f = -beta*y' - sin(y)
beta = 0.1

def f_rhs(t, y, yp):
    return -beta*yp - tf.sin(y)

# -------------------- terminal conditions y^{(k)}(b) = c_k --------------------
terminal_conditions = {
    0: 0.4,   # y(b)  = 0.4
    1: 0.0    # y'(b) = 0.0
}

# -------------------- LGL nodes on [a,b] --------------------
def lgl_nodes(N):
    assert N >= 2
    if N == 2:
        x = np.array([-1.0, 1.0])
    else:
        coeffs = np.zeros(N); coeffs[-1] = 1.0   # P_{N-1}
        dcoeffs = legder(coeffs)                 # d/dx P_{N-1}
        x = np.hstack(([-1.0], np.sort(legroots(dcoeffs)), [1.0]))
    t = 0.5*(x + 1.0)*(b - a) + a
    return t.reshape(-1,1)

# -------------------- Legendre feature layer --------------------
@tf.function
def legendre_polys(xi, M):
    Nbatch = tf.shape(xi)[0]
    ta = tf.TensorArray(xi.dtype, size=M, element_shape=(None,1), clear_after_read=False)
    P0 = tf.ones_like(xi); ta = ta.write(0, P0)
    if M > 1:
        P1 = xi; ta = ta.write(1, P1)
        i = tf.constant(2)
        def cond(i, *_): return i < M
        def body(i, ta, Pim1, Pim2):
            mi = tf.cast(i, xi.dtype)
            Pi = ((2.0*mi - 1.0)*xi*Pim1 - (mi - 1.0)*Pim2) / mi
            ta = ta.write(i, Pi)
            return i+1, ta, Pi, Pim1
        _, ta, _, _ = tf.while_loop(cond, body, [i, ta, P1, P0], parallel_iterations=1)
    stacked = ta.stack()                      # [M, N, 1]
    stacked = tf.transpose(stacked, [1,0,2])  # [N, M, 1]
    return tf.reshape(stacked, [Nbatch, M])   # [N, M]

# -------------------- LNN model (Legendre -> 2*tanh -> y) --------------------
class LNN(tf.keras.Model):
    def __init__(self, M=8, n_hid=64):
        super().__init__()
        self.M = M
        self.fc1 = tf.keras.layers.Dense(n_hid, activation='tanh', dtype='float64')
        self.fc2 = tf.keras.layers.Dense(n_hid, activation='tanh', dtype='float64')
        self.out = tf.keras.layers.Dense(1, dtype='float64')
    
    def call(self, t):
        t  = tf.cast(t, tf.float64)
        xi = (2.0*(t - a)/(b - a)) - 1.0
        L  = legendre_polys(xi, self.M)
        h  = self.fc1(L); h = self.fc2(h)
        return self.out(h)

model = LNN(M=8, n_hid=64)
_ = model(tf.zeros((2,1), dtype=tf.float64))  # build weights

# -------------------- y, y', y'' (2nd-order, graph-safe) --------------------
@tf.function
def y_dy_ddy(t):
    t = tf.cast(t, tf.float64)
    with tf.GradientTape(persistent=True) as g2:
        g2.watch(t)
        with tf.GradientTape() as g1:
            g1.watch(t)
            y = model(t)
        dy = g1.gradient(y, t)
    ddy = g2.gradient(dy, t)
    del g2
    return y, dy, ddy

# -------------------- loss terms --------------------
@tf.function
def loss_terms(t_colloc, t_term, lam_tc):
    y, yp, ypp = y_dy_ddy(t_colloc)
    L_ode = tf.reduce_mean(tf.square(ypp - f_rhs(t_colloc, y, yp)))
    
    yT, ypT, _ = y_dy_ddy(t_term)
    L_tc = tf.constant(0.0, tf.float64)
    if 0 in terminal_conditions:
        L_tc += tf.square(yT  - tf.constant(terminal_conditions[0], tf.float64))
    if 1 in terminal_conditions:
        L_tc += tf.square(ypT - tf.constant(terminal_conditions[1], tf.float64))
    
    return L_ode + lam_tc*L_tc, L_ode, L_tc

# -------------------- collocation --------------------
N = 256
t_colloc = tf.convert_to_tensor(lgl_nodes(N), dtype=tf.float64)
t_term   = tf.constant([[b]], dtype=tf.float64)

# -------------------- training (TC ramp + grad clipping) --------------------
lambda_tc_start = tf.constant(5.0,   dtype=tf.float64)
lambda_tc_final = tf.constant(200.0, dtype=tf.float64)
ramp_epochs     = tf.constant(3000.0, dtype=tf.float64)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
grad_clip = 3.0

@tf.function
def train_step(epoch):
    e     = tf.cast(epoch, tf.float64)
    frac  = e / ramp_epochs
    alpha = tf.minimum(tf.constant(1.0, tf.float64), frac)
    lam   = lambda_tc_start * tf.pow(lambda_tc_final / lambda_tc_start, alpha)
    
    with tf.GradientTape() as tape:
        L, L_ode, L_tc = loss_terms(t_colloc, t_term, lam)
    
    grads = tape.gradient(L, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    if grad_clip:
        grads = [tf.clip_by_norm(g, grad_clip) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L, L_ode, L_tc, lam

def to_scalar(x):
    if hasattr(x,'numpy'): x = x.numpy()
    return np.asarray(x).reshape(()).item()

# -------------------- train --------------------
loss_hist, ode_hist, tc_hist, lam_hist = [], [], [], []
_ = train_step(tf.constant(0, dtype=tf.int32))  # warmup/compile

start = time.time()
epochs = 20000

for epoch in range(epochs):
    L, L_ode, L_tc, lam = train_step(tf.constant(epoch, dtype=tf.int32))
    loss_hist.append(to_scalar(L))
    ode_hist.append(to_scalar(L_ode))
    tc_hist.append(to_scalar(L_tc))
    lam_hist.append(to_scalar(lam))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | L={loss_hist[-1]:.3e} | L_ode={ode_hist[-1]:.3e} "
              f"| L_tc={tc_hist[-1]:.3e} | lambda={lam_hist[-1]:.1f} | elapsed={time.time()-start:.1f}s")

print(f"Training done in {time.time()-start:.2f}s, epochs: {len(loss_hist)}")
print(f"Final: L={loss_hist[-1]:.3e}, L_ode={ode_hist[-1]:.3e}, L_tc={tc_hist[-1]:.3e}, lambda={lam_hist[-1]:.1f}")

# -------------------- evaluate PINN --------------------
t_test = np.linspace(a, b, 600, dtype=np.float64)[:,None]
y_pinn = model(tf.convert_to_tensor(t_test)).numpy().squeeze()

# -------------------- reference (forward in s = b - t) --------------------
# y'' + beta*y' + sin(y) = 0  ->  system in s:
#   Y_s = W
#   W_s = beta*W - sin(Y)
# IC at s=0 (t=b): Y(0)=y(b), W(0)=-y'(b)=0
def rhs_s(s, Z):
    Y, W = Z
    return [W, beta*W - np.sin(Y)]

s_eval = np.linspace(0.0, b-a, 600)
sol_s = solve_ivp(rhs_s, (0.0, b-a),
                  y0=[terminal_conditions.get(0, 0.0), -terminal_conditions.get(1, 0.0)],
                  t_eval=s_eval, method='LSODA', rtol=1e-9, atol=1e-11)

if not sol_s.success:
    print("Reference solver reported failure; continuing with PINN only.")
    t_ref = t_test.squeeze()
    y_ref = np.full_like(t_ref, np.nan)
else:
    t_ref = (b - sol_s.t)[::-1]
    y_ref = sol_s.y[0][::-1]

# -------------------- plots --------------------
plt.figure(figsize=(8,5))
plt.plot(t_test.squeeze(), y_pinn, label='PINN')
if np.all(np.isfinite(y_ref)):
    plt.plot(t_ref, y_ref, '--', label='Reference')
plt.xlabel('t'); plt.ylabel('y(t)')
plt.title('TVP with LNN (composite loss): PINN vs Reference')
plt.legend(); plt.tight_layout()
plt.savefig('pinn_vs_ref_lnn_tvp.png', dpi=150)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(loss_hist, label='Total')
plt.plot(ode_hist,  label='ODE')
plt.plot(tc_hist,   label='TC')
plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss (log scale)')
plt.legend(); plt.tight_layout()
plt.savefig('loss_history_lnn_tvp.png', dpi=150)
plt.show()

# -------------------- CSVs (fixed) --------------------
# 1) Results at evaluation grid (t-sized arrays)
if np.all(np.isfinite(y_ref)):
    y_ref_interp = np.interp(t_test.squeeze(), t_ref, y_ref)
    df_results = pd.DataFrame({'t': t_test.squeeze(),
                               'y_pinn': y_pinn,
                               'y_ref': y_ref_interp})
else:
    df_results = pd.DataFrame({'t': t_test.squeeze(),
                               'y_pinn': y_pinn})
df_results.to_csv('results_lnn_tvp.csv', index=False)

# 2) Training history (epoch-sized arrays)
epochs_arr = np.arange(len(loss_hist))
df_hist = pd.DataFrame({'epoch': epochs_arr,
                        'L_total': loss_hist,
                        'L_ode': ode_hist,
                        'L_tc': tc_hist,
                        'lambda': lam_hist})
df_hist.to_csv('training_history_lnn_tvp.csv', index=False)

print("Saved: pinn_vs_ref_lnn_tvp.png, loss_history_lnn_tvp.png, "
      "results_lnn_tvp.csv, training_history_lnn_tvp.csv")