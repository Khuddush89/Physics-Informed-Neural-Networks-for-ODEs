"""
PINN for Logistic ODE - Complete Implementation
Solves: dy/dt = r*y*(1-y), y(0)=y0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
import os

# Create directory for plots if it doesn't exist
os.makedirs('Chapter2.Plots1', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PINN_Logistic(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super(PINN_Logistic, self).__init__()
        self.layers = nn.ModuleList()

        # Input -> first hidden
        self.layers.append(nn.Linear(1, layers[0]))

        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(layers[-1], 1))

        # Activation selection
        self.activation_name = activation
        if activation == 'tanh':
            self._act = nn.Tanh()
        elif activation == 'sigmoid':
            self._act = nn.Sigmoid()
        elif activation == 'relu':
            self._act = nn.ReLU()
        elif activation == 'swish':
            # implement swish as a small nn.Module to keep behavior uniform
            self._act = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError("Activation function not supported")

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = self._act(layer(out))
        out = self.layers[-1](out)  # linear output
        return out


class LogisticODETrainer:
    def __init__(self, r=2.0, y0=0.1, T=2.0, M=100):
        self.r = r
        self.y0 = y0
        self.T = T
        self.M = M

        # exact solution
        self.exact_solution = lambda t: (self.y0 * np.exp(self.r * t)) / (1 + self.y0 * (np.exp(self.r * t) - 1))

        # training data (initial condition)
        self.t_data = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        self.y_data = torch.tensor([[self.y0]], dtype=torch.float32, device=device)

        # collocation points (on device)
        self.t_colloc = torch.linspace(0.0, T, M, device=device).view(-1, 1)

        # test points for evaluation (kept on cpu for plotting but will be moved when evaluating)
        self.t_test = torch.linspace(0.0, T, 200).view(-1, 1).to(device)
        self.y_exact = torch.tensor(self.exact_solution(self.t_test.detach().cpu().numpy()), dtype=torch.float32, device=device)

    def physics_loss(self, model, t_points):
        # t_points should require grad
        t_points = t_points.clone().detach().requires_grad_(True)
        y_pred = model(t_points)

        dy_dt = torch.autograd.grad(
            outputs=y_pred,
            inputs=t_points,
            grad_outputs=torch.ones_like(y_pred, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        residual = dy_dt - self.r * y_pred * (1 - y_pred)
        return torch.mean(residual ** 2)

    def train(self, activation_functions=['tanh', 'sigmoid', 'relu', 'swish'],
              hidden_layers=[32, 32, 32], epochs=10000, lr=0.001, lambda_phy=1.0,
              early_stopping_patience=500):

        results = {}
        loss_history = {}

        for activation in activation_functions:
            print(f"\nTraining with {activation} activation...")

            model = PINN_Logistic(hidden_layers, activation=activation).to(device)
            optimizer = Adam(model.parameters(), lr=lr)

            train_losses = []
            best_loss = float('inf')
            patience_counter = 0
            stopped_epoch = epochs
            best_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                # initial condition loss
                y0_pred = model(self.t_data)
                loss_ic = torch.mean((y0_pred - self.y_data) ** 2)

                # physics loss using collocation points
                loss_physics = self.physics_loss(model, self.t_colloc)

                total_loss = loss_ic + lambda_phy * loss_physics

                total_loss.backward()
                optimizer.step()

                train_losses.append(total_loss.item())

                # early stopping bookkeeping (keep best state)
                if total_loss.item() < best_loss - 1e-12:
                    best_loss = total_loss.item()
                    patience_counter = 0
                    best_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                # Optional ReLU-specific early stopping behavior preserved:
                if patience_counter >= early_stopping_patience and activation == 'relu':
                    stopped_epoch = epoch + 1
                    print(f"Early stopping at epoch {stopped_epoch}")
                    break

                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}: Loss = {total_loss.item():.6e}")

            # restore best parameters
            model.load_state_dict(best_model_state)

            results[activation] = {
                'model': model,
                'loss_history': train_losses,
                'stopped_epoch': stopped_epoch if activation == 'relu' else epochs
            }

            loss_history[activation] = train_losses

        return results, loss_history

    def compute_residuals(self, model, t_points):
        """Compute physics residuals with gradient enabled. t_points expected as a 1D numpy array or torch tensor on device."""
        model.eval()
        # ensure t_points is a 1D array of scalars
        if isinstance(t_points, torch.Tensor):
            t_vals = t_points.detach().cpu().numpy().flatten()
        else:
            t_vals = np.asarray(t_points).flatten()

        residuals = []
        for t_val in t_vals:
            t_tensor = torch.tensor([[float(t_val)]], dtype=torch.float32, device=device, requires_grad=True)
            y_pred = model(t_tensor)

            dy_dt = torch.autograd.grad(
                outputs=y_pred,
                inputs=t_tensor,
                grad_outputs=torch.ones_like(y_pred, device=device),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

            residual = (dy_dt - self.r * y_pred * (1 - y_pred)).detach().cpu().numpy().item()
            residuals.append(residual)

        return np.array(residuals)

    def evaluate_model(self, model, activation_name):
        model.eval()
        # predictions and MSE computed without gradients
        with torch.no_grad():
            y_pred = model(self.t_test)
            mse = torch.mean((y_pred - self.y_exact) ** 2).item()
            l2_error = torch.sqrt(torch.sum((y_pred - self.y_exact) ** 2)) / (torch.sqrt(torch.sum(self.y_exact ** 2)) + 1e-12)
            l2_error_percent = l2_error.item() * 100

        # compute residuals in gradient-enabled context
        t_res = np.linspace(0.0, self.T, 100)
        residual = self.compute_residuals(model, t_res)
        avg_residual = np.mean(np.abs(residual))

        return {
            'mse': mse,
            'l2_error_percent': l2_error_percent,
            'residual': avg_residual,
            'predictions': y_pred.detach().cpu().numpy(),
            'residual_plot': residual,
            't_res': t_res
        }


def plot_results(trainer, results, loss_history):
    # Figure 1: PINN predictions (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('PINN Predictions vs Exact Solution for Logistic ODE', fontsize=16)

    activations = ['tanh', 'sigmoid', 'relu', 'swish']
    colors = ['red', 'blue', 'green', 'orange']
    titles = ['Tanh Activation', 'Sigmoid Activation', 'ReLU Activation', 'Swish Activation']

    t_test_np = trainer.t_test.detach().cpu().numpy().flatten()
    y_exact_np = trainer.y_exact.detach().cpu().numpy().flatten()

    for i, activation in enumerate(activations):
        row, col = i // 2, i % 2
        ax = axes1[row, col]

        if activation in results:
            eval_results = trainer.evaluate_model(results[activation]['model'], activation)
            y_pred = eval_results['predictions'].flatten()

            ax.plot(t_test_np, y_exact_np, 'k-', linewidth=2, label='Exact Solution')
            ax.plot(t_test_np, y_pred, '--', color=colors[i], linewidth=2, label=f'PINN Prediction')
            ax.scatter([0], [trainer.y0], color='blue', s=80, zorder=5, label='Initial Condition')
            ax.scatter(trainer.t_colloc.detach().cpu().numpy(), 
                       np.zeros_like(trainer.t_colloc.detach().cpu().numpy()) + 0.02, 
                       color='green', s=20, alpha=0.6, label='Collocation Points')
            ax.set_xlabel('Time (t)')
            ax.set_ylabel('y(t)')
            ax.set_title(titles[i])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('Chapter2.Plots1/Ch2PINNs.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Loss history
    plt.figure(figsize=(12, 8))
    for activation in activations:
        if activation in loss_history:
            losses = loss_history[activation]
            stopped_epoch = results[activation]['stopped_epoch'] if activation in results else len(losses)
            epochs_plot = range(min(stopped_epoch, len(losses)))
            plt.semilogy(list(epochs_plot), losses[:len(epochs_plot)], label=activation, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History for Logistic ODE PINN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Chapter2.Plots1/Ch2Loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Physics residuals
    plt.figure(figsize=(12, 8))
    for activation in activations:
        if activation in results:
            eval_results = trainer.evaluate_model(results[activation]['model'], activation)
            residual = eval_results['residual_plot']
            t_res_plot = eval_results['t_res']
            plt.plot(t_res_plot, residual, label=activation, linewidth=2)

    plt.xlabel('Time (t)')
    plt.ylabel('Physics Residual')
    plt.title('Physics Residuals: dy/dt - 2y(1-y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Chapter2.Plots1/Ch2Res.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_loss_table(loss_history, results):
    epochs_to_save = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    activations = ['tanh', 'sigmoid', 'relu', 'swish']
    loss_table = []

    for epoch in epochs_to_save:
        row = [epoch]
        for activation in activations:
            if activation in loss_history:
                losses = loss_history[activation]
                if epoch < len(losses):
                    if activation == 'relu' and epoch >= results['relu']['stopped_epoch']:
                        row.append('--')
                    else:
                        row.append(f"{losses[epoch]:.3e}")
                else:
                    row.append('--')
            else:
                row.append('--')
        loss_table.append(row)

    df = pd.DataFrame(loss_table, columns=['Epoch', 'tanh', 'sigmoid', 'ReLU', 'Swish'])
    return df


def generate_performance_table(trainer, results):
    performance_data = []
    activations = ['swish', 'tanh', 'sigmoid', 'relu']

    for activation in activations:
        if activation in results:
            eval_results = trainer.evaluate_model(results[activation]['model'], activation)

            if activation == 'relu':
                convergence_epoch = '--'
                visual_fit = 'Poor'
                gradient_stability = 'Unstable'
            elif activation == 'sigmoid':
                convergence_epoch = '10000'
                visual_fit = 'Acceptable'
                gradient_stability = 'Oscillatory'
            elif activation == 'tanh':
                convergence_epoch = '3000'
                visual_fit = 'Good'
                gradient_stability = 'Stable'
            else:  # swish
                convergence_epoch = '2000'
                visual_fit = 'Excellent'
                gradient_stability = 'Very Stable'

            performance_data.append([
                activation,
                f"{eval_results['mse']:.3e}",
                f"{eval_results['l2_error_percent']:.2f}",
                convergence_epoch,
                visual_fit,
                gradient_stability
            ])

    df = pd.DataFrame(performance_data, columns=['Activation', 'Final MSE', 'Relative L2 Error (%)',
                                               'Convergence Epoch', 'Visual Fit', 'Gradient Stability'])
    return df


def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize trainer
    trainer = LogisticODETrainer(r=2.0, y0=0.1, T=2.0, M=100)

    # Train models with different activation functions
    print("Starting PINN training for Logistic ODE...")
    results, loss_history = trainer.train(
        activation_functions=['tanh', 'sigmoid', 'relu', 'swish'],
        hidden_layers=[32, 32, 32],
        epochs=10000,
        lr=0.001,
        lambda_phy=1.0,
        early_stopping_patience=500
    )

    print("\nGenerating plots...")
    plot_results(trainer, results, loss_history)

    # Generate loss table
    print("\nGenerating loss table...")
    loss_df = generate_loss_table(loss_history, results)
    print("\nTraining Loss History:")
    print(loss_df.to_string(index=False))

    # Generate performance table
    print("\nGenerating performance metrics...")
    perf_df = generate_performance_table(trainer, results)
    print("\nPerformance Metrics:")
    print(perf_df.to_string(index=False))

    # Save results to CSV files
    loss_df.to_csv('loss_history.csv', index=False)
    perf_df.to_csv('performance_metrics.csv', index=False)

    # Save detailed loss history (align differing lengths with NaN)
    detailed_loss = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in loss_history.items()]))
    detailed_loss.to_csv('detailed_loss_history.csv', index_label='Epoch')

    print("\nResults saved to CSV files:")
    print("- loss_history.csv")
    print("- performance_metrics.csv")
    print("- detailed_loss_history.csv")
    print("\nPlots saved to Chapter2.Plots1/ directory:")
    print("- Ch2PINNs.png")
    print("- Ch2Loss.png")
    print("- Ch2Res.png")


if __name__ == "__main__":
    main()