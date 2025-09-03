#!/usr/bin/env python3
"""
Interactive 3D Optimizer Visualization for Google Colab
======================================================

A complete, self-contained script for visualizing optimizer behavior on 3D loss surfaces
with interactive controls. Copy this entire script into a Google Colab cell.

Features:
- Multiple optimizer implementations (Adam, SGD, AdamW, Muon)
- 3D interactive surface plots with trajectory visualization  
- Interactive widgets for real-time parameter adjustment
- Multiple loss functions (Rosenbrock, Quadratic, Rastrigin)
- Animated optimization paths
- Side-by-side comparisons

Usage in Google Colab:
1. Copy this entire script into a cell
2. Run the cell
3. Use the interactive widgets to explore different optimizers and parameters
"""

# ===========================
# IMPORTS AND SETUP
# ===========================
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Colab-specific setup
try:
    from google.colab import output
    IN_COLAB = True
    print("🚀 Running in Google Colab - interactive widgets enabled!")
except ImportError:
    IN_COLAB = False
    print("📊 Running locally - some interactive features may be limited")

# ===========================
# OPTIMIZER IMPLEMENTATIONS
# ===========================

class AdamState(NamedTuple):
    """Adam optimizer state"""
    step: int
    m: jnp.ndarray  # First moment
    v: jnp.ndarray  # Second moment

def adam_init(params: jnp.ndarray, **kwargs) -> AdamState:
    """Initialize Adam optimizer state"""
    return AdamState(step=0, m=jnp.zeros_like(params), v=jnp.zeros_like(params))

def adam_update(grads: jnp.ndarray, state: AdamState, params: jnp.ndarray, 
                lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, 
                eps: float = 1e-8) -> Tuple[jnp.ndarray, AdamState]:
    """Adam optimizer update step"""
    step = state.step + 1
    m = beta1 * state.m + (1 - beta1) * grads
    v = beta2 * state.v + (1 - beta2) * grads**2
    
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    updates = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
    new_state = AdamState(step=step, m=m, v=v)
    return updates, new_state

class SGDState(NamedTuple):
    """SGD optimizer state"""
    step: int
    momentum: jnp.ndarray

def sgd_init(params: jnp.ndarray, **kwargs) -> SGDState:
    """Initialize SGD optimizer state"""
    return SGDState(step=0, momentum=jnp.zeros_like(params))

def sgd_update(grads: jnp.ndarray, state: SGDState, params: jnp.ndarray,
               lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0,
               nesterov: bool = False) -> Tuple[jnp.ndarray, SGDState]:
    """SGD optimizer update step with momentum"""
    step = state.step + 1
    
    # Apply weight decay
    if weight_decay > 0:
        grads = grads + weight_decay * params
    
    # Update momentum
    new_momentum = momentum * state.momentum + grads
    
    # Apply update
    if nesterov:
        updates = -lr * (momentum * new_momentum + grads)
    else:
        updates = -lr * new_momentum
    
    new_state = SGDState(step=step, momentum=new_momentum)
    return updates, new_state

class AdamWState(NamedTuple):
    """AdamW optimizer state"""
    step: int
    m: jnp.ndarray
    v: jnp.ndarray

def adamw_init(params: jnp.ndarray, **kwargs) -> AdamWState:
    """Initialize AdamW optimizer state"""
    return AdamWState(step=0, m=jnp.zeros_like(params), v=jnp.zeros_like(params))

def adamw_update(grads: jnp.ndarray, state: AdamWState, params: jnp.ndarray,
                 lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01) -> Tuple[jnp.ndarray, AdamWState]:
    """AdamW optimizer update step with decoupled weight decay"""
    step = state.step + 1
    
    # Adam updates (without weight decay in gradients)
    m = beta1 * state.m + (1 - beta1) * grads
    v = beta2 * state.v + (1 - beta2) * grads**2
    
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    # AdamW: apply weight decay directly to parameters
    adam_updates = -lr * m_hat / (jnp.sqrt(v_hat) + eps)
    weight_decay_updates = -lr * weight_decay * params
    updates = adam_updates + weight_decay_updates
    
    new_state = AdamWState(step=step, m=m, v=v)
    return updates, new_state

# ===========================
# LOSS FUNCTIONS
# ===========================

class LossFunction:
    """Base class for loss functions"""
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, params: jnp.ndarray) -> float:
        raise NotImplementedError
    
    def grad(self, params: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(self.__call__)(params)

class RosenbrockFunction(LossFunction):
    """Rosenbrock banana function - classic optimization challenge"""
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__(f"Rosenbrock (a={a}, b={b})")
        self.a = a
        self.b = b
    
    def __call__(self, params: jnp.ndarray) -> float:
        x, y = params[0], params[1]
        return (self.a - x)**2 + self.b * (y - x**2)**2

class QuadraticFunction(LossFunction):
    """Quadratic bowl function"""
    def __init__(self, condition_number: float = 10.0):
        super().__init__(f"Quadratic (cond={condition_number})")
        self.condition_number = condition_number
        # Create condition number matrix
        eigenvals = jnp.array([1.0, condition_number])
        self.A = jnp.diag(eigenvals)
    
    def __call__(self, params: jnp.ndarray) -> float:
        return 0.5 * params.T @ self.A @ params

class RastriginFunction(LossFunction):
    """Rastrigin function - highly multi-modal"""
    def __init__(self, A: float = 10.0):
        super().__init__(f"Rastrigin (A={A})")
        self.A = A
    
    def __call__(self, params: jnp.ndarray) -> float:
        x, y = params[0], params[1]
        return (self.A * 2 + x**2 - self.A * jnp.cos(2 * jnp.pi * x) + 
                y**2 - self.A * jnp.cos(2 * jnp.pi * y))

class SaddleFunction(LossFunction):
    """Simple saddle point function"""
    def __init__(self):
        super().__init__("Saddle Point")
    
    def __call__(self, params: jnp.ndarray) -> float:
        x, y = params[0], params[1]
        return x**2 - y**2

# ===========================
# OPTIMIZATION TRACKING
# ===========================

def run_optimization(optimizer_init, optimizer_update, loss_fn, initial_params, 
                    num_steps: int = 200, **optimizer_kwargs):
    """Run optimization and track trajectory"""
    params = initial_params.copy()
    state = optimizer_init(params, **optimizer_kwargs)
    
    trajectory = {
        'params': [params.copy()],
        'losses': [float(loss_fn(params))],
        'gradients': [],
        'updates': []
    }
    
    for step in range(num_steps):
        grads = loss_fn.grad(params)
        updates, state = optimizer_update(grads, state, params, **optimizer_kwargs)
        
        params = params + updates
        loss_val = float(loss_fn(params))
        
        trajectory['params'].append(params.copy())
        trajectory['losses'].append(loss_val)
        trajectory['gradients'].append(grads.copy())
        trajectory['updates'].append(updates.copy())
        
        # Early stopping if converged
        if loss_val < 1e-6:
            break
    
    return trajectory

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def create_3d_surface(loss_fn, x_range=(-3, 3), y_range=(-3, 3), resolution=50):
    """Create 3D surface data for loss function"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            params = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = loss_fn(params)
    
    return X, Y, Z

def plot_interactive_3d(loss_fn, trajectories=None, title="Loss Surface"):
    """Create interactive 3D plot with Plotly"""
    X, Y, Z = create_3d_surface(loss_fn, resolution=40)
    
    # Create surface plot
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='viridis',
        opacity=0.7,
        name='Loss Surface',
        showscale=True
    ))
    
    # Add optimizer trajectories
    if trajectories:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, traj) in enumerate(trajectories.items()):
            params_array = np.array(traj['params'])
            losses_array = np.array(traj['losses'])
            
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=params_array[:, 0],
                y=params_array[:, 1], 
                z=losses_array,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=4),
                marker=dict(size=3),
                name=name
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter3d(
                x=[params_array[0, 0]], 
                y=[params_array[0, 1]], 
                z=[losses_array[0]],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)], symbol='circle'),
                name=f'{name} Start',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[params_array[-1, 0]], 
                y=[params_array[-1, 1]], 
                z=[losses_array[-1]],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)], symbol='square'),
                name=f'{name} End',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Interactive 3D Visualization: {title}",
        scene=dict(
            xaxis_title="Parameter X",
            yaxis_title="Parameter Y", 
            zaxis_title="Loss Value",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig

def plot_loss_curves(trajectories, title="Loss Curves"):
    """Plot loss curves over time"""
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, traj) in enumerate(trajectories.items()):
        fig.add_trace(go.Scatter(
            x=list(range(len(traj['losses']))),
            y=traj['losses'],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=name
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Optimization Step",
        yaxis_title="Loss Value",
        yaxis_type="log",
        width=800,
        height=400,
        showlegend=True
    )
    
    return fig

def create_contour_plot(loss_fn, trajectories=None, x_range=(-3, 3), y_range=(-3, 3)):
    """Create 2D contour plot with trajectories"""
    X, Y, Z = create_3d_surface(loss_fn, x_range, y_range, resolution=50)
    
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=np.linspace(x_range[0], x_range[1], 50),
        y=np.linspace(y_range[0], y_range[1], 50),
        z=Z,
        colorscale='viridis',
        opacity=0.6,
        contours=dict(coloring='heatmap'),
        showscale=True,
        name='Loss Contours'
    ))
    
    # Add trajectories
    if trajectories:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, traj) in enumerate(trajectories.items()):
            params_array = np.array(traj['params'])
            
            fig.add_trace(go.Scatter(
                x=params_array[:, 0],
                y=params_array[:, 1],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=4, color=colors[i % len(colors)]),
                name=name
            ))
            
            # Mark start and end
            fig.add_trace(go.Scatter(
                x=[params_array[0, 0]], 
                y=[params_array[0, 1]],
                mode='markers',
                marker=dict(size=10, color=colors[i % len(colors)], symbol='circle'),
                name=f'{name} Start',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[params_array[-1, 0]], 
                y=[params_array[-1, 1]],
                mode='markers', 
                marker=dict(size=10, color=colors[i % len(colors)], symbol='square'),
                name=f'{name} End',
                showlegend=False
            ))
    
    fig.update_layout(
        title="2D Trajectory View",
        xaxis_title="Parameter X",
        yaxis_title="Parameter Y",
        width=600,
        height=600,
        showlegend=True
    )
    
    return fig

# ===========================
# INTERACTIVE WIDGETS
# ===========================

class OptimizerComparison:
    """Main class for interactive optimizer comparison"""
    
    def __init__(self):
        self.optimizers = {
            'Adam': (adam_init, adam_update, {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999}),
            'SGD': (sgd_init, sgd_update, {'lr': 0.01, 'momentum': 0.9}),
            'AdamW': (adamw_init, adamw_update, {'lr': 0.01, 'weight_decay': 0.01})
        }
        
        self.loss_functions = {
            'Rosenbrock': RosenbrockFunction(),
            'Quadratic': QuadraticFunction(condition_number=10.0),
            'Rastrigin': RastriginFunction(),
            'Saddle': SaddleFunction()
        }
        
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup interactive widgets"""
        # Loss function selection
        self.loss_dropdown = widgets.Dropdown(
            options=list(self.loss_functions.keys()),
            value='Rosenbrock',
            description='Loss Function:'
        )
        
        # Optimizer selection
        self.optimizer_checks = widgets.VBox([
            widgets.Checkbox(value=True, description='Adam'),
            widgets.Checkbox(value=True, description='SGD'), 
            widgets.Checkbox(value=False, description='AdamW')
        ])
        
        # Parameters
        self.lr_slider = widgets.FloatLogSlider(
            value=0.01, min=-4, max=-1, step=0.1,
            description='Learning Rate:', readout_format='.4f'
        )
        
        self.steps_slider = widgets.IntSlider(
            value=200, min=50, max=1000, step=50,
            description='Optimization Steps:'
        )
        
        self.initial_x = widgets.FloatSlider(
            value=-2.0, min=-3.0, max=3.0, step=0.1,
            description='Initial X:'
        )
        
        self.initial_y = widgets.FloatSlider(
            value=2.0, min=-3.0, max=3.0, step=0.1,
            description='Initial Y:'
        )
        
        # Run button
        self.run_button = widgets.Button(
            description='🚀 Run Optimization',
            button_style='primary'
        )
        
        self.run_button.on_click(self.run_optimization)
        
        # Output area
        self.output = widgets.Output()
        
        # Layout
        controls_left = widgets.VBox([
            self.loss_dropdown,
            widgets.HTML("<b>Select Optimizers:</b>"),
            self.optimizer_checks,
        ])
        
        controls_right = widgets.VBox([
            self.lr_slider,
            self.steps_slider,
            self.initial_x,
            self.initial_y,
            self.run_button
        ])
        
        self.controls = widgets.HBox([controls_left, controls_right])
        
        # Display
        self.widget = widgets.VBox([
            widgets.HTML("<h2>🎯 Interactive Optimizer Comparison</h2>"),
            self.controls,
            self.output
        ])
    
    def run_optimization(self, b=None):
        """Run optimization with current parameters"""
        with self.output:
            self.output.clear_output(wait=True)
            print("🔄 Running optimization...")
            
            # Get selected parameters
            loss_fn = self.loss_functions[self.loss_dropdown.value]
            initial_params = jnp.array([self.initial_x.value, self.initial_y.value])
            lr = self.lr_slider.value
            num_steps = self.steps_slider.value
            
            # Get selected optimizers
            selected_optimizers = []
            optimizer_names = ['Adam', 'SGD', 'AdamW']
            for i, checkbox in enumerate(self.optimizer_checks.children):
                if checkbox.value:
                    selected_optimizers.append(optimizer_names[i])
            
            if not selected_optimizers:
                print("❌ Please select at least one optimizer!")
                return
            
            # Run optimizations
            trajectories = {}
            for name in selected_optimizers:
                optimizer_init, optimizer_update, default_kwargs = self.optimizers[name]
                kwargs = default_kwargs.copy()
                kwargs['lr'] = lr
                
                trajectory = run_optimization(
                    optimizer_init, optimizer_update, loss_fn, 
                    initial_params, num_steps, **kwargs
                )
                trajectories[name] = trajectory
                
                final_loss = trajectory['losses'][-1]
                print(f"✅ {name}: Final loss = {final_loss:.6f}")
            
            print(f"🎨 Creating visualizations...")
            
            # Create visualizations
            fig_3d = plot_interactive_3d(loss_fn, trajectories, loss_fn.name)
            fig_contour = create_contour_plot(loss_fn, trajectories)
            fig_losses = plot_loss_curves(trajectories)
            
            # Display plots
            fig_3d.show()
            fig_contour.show()
            fig_losses.show()
            
            print("✨ Optimization complete! Explore the interactive plots above.")
    
    def display(self):
        """Display the widget interface"""
        display(self.widget)

# ===========================
# PRESET EXPERIMENTS
# ===========================

def run_rosenbrock_comparison():
    """Preset: Compare optimizers on Rosenbrock function"""
    print("🍌 Rosenbrock Function Comparison")
    print("="*50)
    
    loss_fn = RosenbrockFunction()
    initial_params = jnp.array([-2.0, 2.0])
    
    optimizers_config = {
        'Adam': (adam_init, adam_update, {'lr': 0.01}),
        'SGD+Momentum': (sgd_init, sgd_update, {'lr': 0.002, 'momentum': 0.9}),
        'AdamW': (adamw_init, adamw_update, {'lr': 0.01, 'weight_decay': 0.001})
    }
    
    trajectories = {}
    for name, (opt_init, opt_update, kwargs) in optimizers_config.items():
        trajectory = run_optimization(opt_init, opt_update, loss_fn, initial_params, 300, **kwargs)
        trajectories[name] = trajectory
        print(f"✅ {name}: Final loss = {trajectory['losses'][-1]:.6f}")
    
    # Create visualizations
    fig_3d = plot_interactive_3d(loss_fn, trajectories, "Rosenbrock Function")
    fig_contour = create_contour_plot(loss_fn, trajectories, x_range=(-2.5, 2.5), y_range=(-1, 3))
    fig_losses = plot_loss_curves(trajectories, "Rosenbrock: Loss Curves")
    
    fig_3d.show()
    fig_contour.show() 
    fig_losses.show()

def run_multi_surface_comparison():
    """Compare Adam across different loss surfaces"""
    print("🎭 Multi-Surface Comparison (Adam Optimizer)")
    print("="*50)
    
    surfaces = {
        'Rosenbrock': RosenbrockFunction(),
        'Quadratic': QuadraticFunction(condition_number=50),
        'Rastrigin': RastriginFunction(),
        'Saddle': SaddleFunction()
    }
    
    initial_positions = {
        'Rosenbrock': jnp.array([-2.0, 2.0]),
        'Quadratic': jnp.array([2.0, 2.0]),
        'Rastrigin': jnp.array([2.5, 2.5]),
        'Saddle': jnp.array([2.0, 0.5])
    }
    
    for name, loss_fn in surfaces.items():
        print(f"\n🎯 Testing on {name}...")
        initial_params = initial_positions[name]
        
        trajectory = run_optimization(adam_init, adam_update, loss_fn, initial_params, 
                                    200, lr=0.01, beta1=0.9, beta2=0.999)
        
        print(f"Final loss: {trajectory['losses'][-1]:.6f}")
        
        # Create visualization
        fig = plot_interactive_3d(loss_fn, {'Adam': trajectory}, f"{name} Surface")
        fig.show()

def create_animation_demo():
    """Create animated optimization demo"""
    print("🎬 Creating Animated Optimization Demo")
    print("="*40)
    
    # Setup
    loss_fn = RosenbrockFunction()
    initial_params = jnp.array([-2.0, 2.0])
    
    # Run optimization
    trajectory = run_optimization(adam_init, adam_update, loss_fn, initial_params, 100, lr=0.01)
    
    # Create animated plot
    fig = plt.figure(figsize=(12, 5))
    
    # 2D contour subplot
    ax1 = fig.add_subplot(121)
    X, Y, Z = create_3d_surface(loss_fn, resolution=50)
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
    ax1.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
    
    # 3D surface subplot  
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    
    def animate(frame):
        if frame < len(trajectory['params']):
            # Clear previous points
            ax1.clear()
            ax2.clear()
            
            # Redraw surfaces
            ax1.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
            ax1.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
            ax2.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
            
            # Draw trajectory up to current frame
            params_so_far = np.array(trajectory['params'][:frame+1])
            losses_so_far = np.array(trajectory['losses'][:frame+1])
            
            if len(params_so_far) > 1:
                ax1.plot(params_so_far[:, 0], params_so_far[:, 1], 'r-', linewidth=2)
                ax2.plot(params_so_far[:, 0], params_so_far[:, 1], losses_so_far, 'r-', linewidth=2)
            
            # Current position
            current_params = trajectory['params'][frame]
            current_loss = trajectory['losses'][frame]
            ax1.plot(current_params[0], current_params[1], 'ro', markersize=8)
            ax2.scatter(current_params[0], current_params[1], current_loss, color='red', s=50)
            
            ax1.set_title(f'2D View (Step {frame})')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            
            ax2.set_title(f'3D View (Loss: {current_loss:.3f})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y') 
            ax2.set_zlabel('Loss')
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory['params']), interval=100, repeat=True)
    plt.tight_layout()
    plt.show()
    
    return anim

# ===========================
# MAIN DEMO FUNCTION
# ===========================

def main_demo():
    """Main demo function - run this in Google Colab"""
    print("🎉 Welcome to Interactive Optimizer Visualization!")
    print("="*60)
    print()
    print("This demo provides multiple ways to explore optimizer behavior:")
    print("1. Interactive widget interface")
    print("2. Preset comparison experiments") 
    print("3. Animated optimization paths")
    print()
    
    # Display interactive widget
    print("🎛️  INTERACTIVE CONTROLS")
    print("-" * 30)
    comparator = OptimizerComparison()
    comparator.display()
    
    print("\n" + "="*60)
    print("📊 PRESET EXPERIMENTS")
    print("Run the functions below to see preset comparisons:")
    print()
    print("• run_rosenbrock_comparison() - Compare all optimizers on Rosenbrock")
    print("• run_multi_surface_comparison() - Test Adam on different surfaces")
    print("• create_animation_demo() - Animated optimization demo")
    print()
    
    return comparator

# ===========================
# COLAB QUICK START
# ===========================

if __name__ == "__main__" or IN_COLAB:
    print("🚀 Quick Start for Google Colab:")
    print("-" * 40)
    print("1. Run this cell to load all functions")
    print("2. Call main_demo() to start the interactive interface")
    print("3. Or run preset experiments directly:")
    print("   • run_rosenbrock_comparison()")
    print("   • run_multi_surface_comparison()")
    print("   • create_animation_demo()")
    print()
    
    # Auto-start the main demo
    if IN_COLAB:
        main_demo()

# Example usage for Google Colab:
"""
# Copy this entire script into a Google Colab cell, then run:

# Start interactive demo
demo = main_demo()

# Or run preset experiments
run_rosenbrock_comparison()
run_multi_surface_comparison() 
anim = create_animation_demo()

# The interactive widgets will appear below the cell output
# Use them to explore different optimizers, loss functions, and parameters in real-time!
"""