# chebyshev_pinn_duffing.py
# Chebyshev Neural Network for Duffing Oscillator
# Downloadable Python file

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define hyperparameters
m = 5  # Number of Chebyshev polynomials
n_hid = 10  # Number of neurons per hidden layer
N = 200  # Increased number of collocation points for better resolution
lambda_bc = 100.0  # Boundary condition penalty parameter
learning_rate = 1e-3  # Learning rate for Adam
adam_epochs = 100000  # Number of Adam epochs
lbfgs_max_iter = 10000  # Maximum L-BFGS iterations
epsilon = 0.1  # Nonlinearity parameter for Duffing oscillator
clip_norm = 1.0  # Gradient clipping norm

# Define the domain
a, b = 0.0, np.pi  # Interval [0, pi]

# Chebyshev-Gauss-Lobatto collocation points
def chebyshev_lobatto_points(N, a, b):
    i = np.arange(N)
    t = (a + b) / 2 + (b - a) / 2 * np.cos(i * np.pi / (N - 1))
    return tf.convert_to_tensor(t, dtype=tf.float32)

# Chebyshev polynomials of the first kind
def chebyshev_polynomials(xi, m):
    xi = tf.reshape(xi, [-1])  # Ensure xi is 1D
    C = [tf.ones_like(xi), xi]  # C_0 = 1, C_1 = xi
    for k in range(2, m):
        C_k = 2 * xi * C[k-1] - C[k-2]  # Recurrence relation
        C.append(C_k)
    return tf.stack(C, axis=1)  # Shape: [batch_size, m]

# Define the Chebyshev Neural Network
class ChNN(tf.keras.Model):
    def __init__(self, m, n_hid):
        super(ChNN, self).__init__()
        self.m = m
        self.n_hid = n_hid
        
        # Initialize weights and biases with Xavier uniform
        initializer = tf.keras.initializers.GlorotUniform()
        
        # First hidden layer: m inputs to n_hid neurons
        self.W1 = self.add_weight(shape=(m, n_hid), initializer=initializer, trainable=True, name='W1')
        self.b1 = self.add_weight(shape=(n_hid,), initializer='zeros', trainable=True, name='b1')
        
        # Second hidden layer: n_hid inputs to n_hid neurons
        self.W2 = self.add_weight(shape=(n_hid, n_hid), initializer=initializer, trainable=True, name='W2')
        self.b2 = self.add_weight(shape=(n_hid,), initializer='zeros', trainable=True, name='b2')
        
        # Output layer: n_hid inputs to 1 output
        self.W3 = self.add_weight(shape=(n_hid, 1), initializer=initializer, trainable=True, name='W3')
        self.b3 = self.add_weight(shape=(1,), initializer='zeros', trainable=True, name='b3')
    
    def call(self, t):
        t = tf.reshape(t, [-1, 1])  # Ensure t is [batch_size, 1]
        xi = (2 * t - (a + b)) / (b - a)  # Scale to [-1, 1]
        C = chebyshev_polynomials(xi, self.m)  # Shape: [batch_size, m]
        z1 = tf.matmul(C, self.W1) + self.b1  # Shape: [batch_size, n_hid]
        h1 = tf.nn.tanh(z1)
        z2 = tf.matmul(h1, self.W2) + self.b2
        h2 = tf.nn.tanh(z2)
        y = tf.matmul(h2, self.W3) + self.b3
        return y

# Compute higher-order derivatives
def compute_derivatives(model, t, order):
    t = tf.reshape(t, [-1, 1])  # Ensure t is [batch_size, 1]
    with tf.GradientTape() as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            y = model(t)
        y_prime = tape1.gradient(y, t) if order >= 1 else None
    y_double_prime = tape2.gradient(y_prime, t) if order >= 2 and y_prime is not None else None
    return [y, y_prime, y_double_prime]

# Define the ODE residual for Duffing oscillator
def ode_residual(t, y, y_prime, y_double_prime):
    if y_double_prime is None:
        print("Warning: Second derivative is None")
        return tf.zeros_like(y)  # Return zero residual
    return y_double_prime + y + epsilon * y**3

# Define a numerical reference solution using SciPy's solve_bvp
def reference_solution(t):
    def fun(t, y):
        return np.vstack((y[1], -y[0] - epsilon * y[0]**3))
    
    def bc(ya, yb):
        return np.array([ya[0] - 1, yb[1]])  # y(0) = 1, y'(pi) = 0
    
    t_ref = np.linspace(a, b, 100)
    y_guess = np.zeros((2, t_ref.size))  # Initial guess: y=0, y'=0
    sol = solve_bvp(fun, bc, t_ref, y_guess, tol=1e-6, max_nodes=1000)
    return sol.sol(t)[0]  # Return y(t)

# Define the loss function
def compute_loss(model, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b):
    # Compute ODE residual loss
    t_colloc = tf.reshape(t_colloc, [-1, 1])
    derivatives = compute_derivatives(model, t_colloc, 2)
    y, y_prime, y_double_prime = derivatives
    
    if y_double_prime is None:
        print("Warning: Second derivative is None in compute_loss")
        return tf.constant(1e10, dtype=tf.float32)
    
    residual = ode_residual(t_colloc, y, y_prime, y_double_prime)
    loss_ode = tf.reduce_mean(tf.square(residual))
    
    # Compute boundary condition loss
    y_a = model(tf.reshape(t_bc_a, [-1, 1]))
    derivatives_b = compute_derivatives(model, tf.reshape(t_bc_b, [-1, 1]), 1)
    y_prime_b = derivatives_b[1]
    if y_prime_b is None:
        print("Warning: y_prime_b is None in compute_loss")
        return tf.constant(1e10, dtype=tf.float32)
    loss_bc = tf.reduce_mean(tf.square(y_a - y_bc_a)) + tf.reduce_mean(tf.square(y_prime_b - y_prime_bc_b))
    
    # Total loss
    return loss_ode + lambda_bc * loss_bc

# Training step for Adam with gradient clipping
@tf.function
def train_step(model, optimizer, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    if any(g is None for g in gradients):
        print("Warning: Some gradients are None")
        print("Trainable variables:", [v.name for v in model.trainable_variables])
        return loss
    
    # Clip gradients to stabilize training
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# L-BFGS optimizer
def lbfgs_train(model, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b, max_iter):
    def loss_and_grads(weights):
        trainable_vars = model.trainable_variables
        shapes = [v.shape.as_list() for v in trainable_vars]
        sizes = [np.prod(s) for s in shapes]
        split_weights = tf.split(weights, sizes)
        
        for var, w in zip(trainable_vars, split_weights):
            var.assign(tf.reshape(w, var.shape))
        
        with tf.GradientTape() as tape:
            loss = compute_loss(model, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b)
        grads = tape.gradient(loss, model.trainable_variables)
        grad_values = [g.numpy().flatten() if g is not None else np.zeros_like(v).flatten() for g, v in zip(grads, model.trainable_variables)]
        return loss.numpy(), np.concatenate(grad_values)
    
    from scipy.optimize import minimize
    initial_weights = np.concatenate([v.numpy().flatten() for v in model.trainable_variables])
    result = minimize(loss_and_grads, initial_weights, method='L-BFGS-B', jac=True, options={'maxiter': max_iter})
    
    split_weights = tf.split(result.x, [np.prod(v.shape) for v in model.trainable_variables])
    for var, w in zip(model.trainable_variables, split_weights):
        var.assign(tf.reshape(w, var.shape))
    
    return result.fun

# Main training function
def train_pinn():
    # Initialize model
    model = ChNN(m, n_hid)
    
    # Generate collocation points
    t_colloc = chebyshev_lobatto_points(N, a, b)
    
    # Define boundary conditions
    t_bc_a = tf.constant([a], dtype=tf.float32)  # t = 0
    t_bc_b = tf.constant([b], dtype=tf.float32)  # t = pi
    y_bc_a = tf.constant([1.0], dtype=tf.float32)  # y(0) = 1
    y_prime_bc_b = tf.constant([0.0], dtype=tf.float32)  # y'(pi) = 0
    
    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # Track loss history
    loss_history = []
    
    print("Starting Adam training...")
    
    # Training loop
    for epoch in range(adam_epochs):
        loss = train_step(model, optimizer, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b)
        loss_history.append(loss.numpy())
        if epoch % 500 == 0:  # Log every 500 epochs
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
        
        # Check convergence over a window of 10 epochs
        if epoch > 1000 and len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 1e-6 for i in range(1, len(recent_losses))):
                print(f"Converged at epoch {epoch}")
                break
    
    print("Adam training completed. Starting L-BFGS fine-tuning...")
    
    # L-BFGS fine-tuning
    lbfgs_loss = lbfgs_train(model, t_colloc, t_bc_a, t_bc_b, y_bc_a, y_prime_bc_b, lbfgs_max_iter)
    print(f"L-BFGS final loss: {lbfgs_loss:.6f}")
    loss_history.append(lbfgs_loss)
    
    return model, loss_history

# Plotting and comparison function
def plot_results(model, loss_history):
    # Generate test points for evaluation
    t_test = tf.linspace(a, b, 100, axis=0)
    t_test = tf.reshape(t_test, [-1, 1])
    y_pred = model(t_test)
    y_ref = reference_solution(t_test.numpy())  # Numerical reference solution
    y_ref = tf.convert_to_tensor(y_ref, dtype=tf.float32)
    
    # Compute metrics
    l2_error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_ref)))
    mae = tf.reduce_max(tf.abs(y_pred - y_ref))
    
    # Compute boundary condition errors
    y_a = model(tf.reshape(tf.constant([a], dtype=tf.float32), [-1, 1]))
    bc_error_y0 = tf.abs(y_a - 1.0).numpy().item()
    derivatives_b = compute_derivatives(model, tf.reshape(tf.constant([b], dtype=tf.float32), [-1, 1]), 1)
    y_prime_b = derivatives_b[1]
    bc_error_yprime_pi = tf.abs(y_prime_b).numpy().item() if y_prime_b is not None else float('inf')
    
    # Compute ODE residual error
    derivatives_test = compute_derivatives(model, t_test, 2)
    y_test, y_prime_test, y_double_prime_test = derivatives_test
    residual = ode_residual(t_test, y_test, y_prime_test, y_double_prime_test)
    residual_error = tf.reduce_mean(tf.abs(residual))
    
    print(f"L2 error: {l2_error.numpy().item():.6f}")
    print(f"Maximum Absolute Error: {mae.numpy().item():.6f}")
    print(f"Boundary condition error at y(0): {bc_error_y0:.6f}")
    print(f"Boundary condition error at y'(pi): {bc_error_yprime_pi:.6f}")
    print(f"Mean ODE residual error: {residual_error.numpy().item():.6f}")
    
    # Create three separate figures
    # Figure 1: PINN vs Reference Solution
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, y_pred, 'b-', label='PINN Solution', linewidth=2)
    plt.plot(t_test, y_ref, 'g--', label='Reference Solution', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('PINN vs Reference Solution for Duffing Oscillator')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Training Loss History with moving average
    window_size = 50
    moving_avg = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history, 'r-', alpha=0.3, label='Raw Loss')
    plt.plot(range(window_size-1, len(loss_history)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.yscale('log')  # Log scale for better visualization
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Pointwise Error
    pointwise_error = tf.abs(y_pred - y_ref)
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, pointwise_error, 'm-', label='|PINN - Reference|', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Pointwise Error')
    plt.title('Pointwise Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'l2_error': l2_error.numpy().item(),
        'max_abs_error': mae.numpy().item(),
        'bc_error_y0': bc_error_y0,
        'bc_error_yprime_pi': bc_error_yprime_pi,
        'residual_error': residual_error.numpy().item()
    }

# Save results to CSV files
def save_results_to_csv(model, loss_history):
    # Generate test points
    t_test = tf.linspace(a, b, 1000, axis=0)
    t_test = tf.reshape(t_test, [-1, 1])
    y_pred = model(t_test)
    y_ref = reference_solution(t_test.numpy())
    
    # Save predictions
    predictions = np.column_stack([t_test.numpy().flatten(), y_pred.numpy().flatten(), y_ref])
    np.savetxt('duffing_predictions.csv', predictions, delimiter=',', 
               header='t,y_pred,y_ref', comments='')
    
    # Save loss history
    loss_data = np.column_stack([range(len(loss_history)), loss_history])
    np.savetxt('duffing_loss_history.csv', loss_data, delimiter=',', 
               header='epoch,loss', comments='')
    
    # Compute and save residuals
    derivatives = compute_derivatives(model, t_test, 2)
    y, y_prime, y_double_prime = derivatives
    residuals = ode_residual(t_test, y, y_prime, y_double_prime)
    
    residual_data = np.column_stack([t_test.numpy().flatten(), residuals.numpy().flatten()])
    np.savetxt('duffing_residuals.csv', residual_data, delimiter=',', 
               header='t,residual', comments='')
    
    print("Results saved to CSV files:")
    print("- duffing_predictions.csv")
    print("- duffing_loss_history.csv")
    print("- duffing_residuals.csv")

# Main execution
if __name__ == "__main__":
    print("Chebyshev Neural Network for Duffing Oscillator")
    print("=" * 50)
    print(f"Parameters: m={m}, n_hid={n_hid}, N={N}, epsilon={epsilon}")
    print(f"Domain: [{a}, {b}]")
    print(f"Boundary conditions: y(0)=1, y'(π)=0")
    print()
    
    # Train the model
    model, loss_history = train_pinn()
    
    # Plot results
    print("\nPlotting results...")
    metrics = plot_results(model, loss_history)
    
    # Save results to CSV
    print("\nSaving results to CSV files...")
    save_results_to_csv(model, loss_history)
    
    print("\nTraining completed successfully!")
    print("Metrics summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")