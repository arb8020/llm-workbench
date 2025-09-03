#!/usr/bin/env python3
"""
Terminal Optimizer Demo

Beautiful terminal-based optimizer comparison using plotext and rich.
No matplotlib windows - everything renders in your terminal!
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
from typing import Dict, List

# Terminal plotting libraries
try:
    import plotext as plt  # Terminal plotting
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    TERMINAL_LIBS_AVAILABLE = True
except ImportError:
    TERMINAL_LIBS_AVAILABLE = False
    print("⚠️  Install terminal plotting libraries:")
    print("   pip install plotext rich")
    sys.exit(1)

# Setup paths
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..', 'validation'))

# Import optimizers
from adam.solution import adam_init, adam_update
from adamw.solution import adamw_init, adamw_update
from sgd.solution import sgd_init, sgd_update
from muon.solution import muon_init, muon_update

# Import tools
from contract import RosenbrockModel, SimpleQuadraticModel, OptimizerConfig
from compare import compare_optimizers, analyze_convergence

console = Console()


class TerminalOptimizer:
    def __init__(self):
        self.optimizers = {
            'Adam': OptimizerConfig('Adam', adam_init, adam_update, {'lr': 0.01}, 'red'),
            'AdamW': OptimizerConfig('AdamW', adamw_init, adamw_update, {'lr': 0.01, 'weight_decay': 0.001}, 'blue'),
            'SGD': OptimizerConfig('SGD+Mom', sgd_init, sgd_update, {'lr': 0.002, 'momentum': 0.9}, 'green'),
            'Muon': OptimizerConfig('Muon', muon_init, muon_update, {'lr': 0.005, 'momentum': 0.9}, 'purple')
        }
        
    def plot_loss_curves(self, results: Dict, title: str = "Loss Curves"):
        """Plot loss curves in terminal"""
        
        plt.clf()  # Clear previous plot
        plt.title(title)
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss (log scale)")
        
        colors = ['red', 'blue', 'green', 'magenta', 'cyan']
        
        for i, (name, trajectory) in enumerate(results.items()):
            losses = np.array(trajectory.losses)
            steps = np.arange(len(losses))
            
            # Use log scale for better visualization
            log_losses = np.log10(losses + 1e-10)
            
            plt.plot(steps, log_losses, label=name, color=colors[i % len(colors)])
        
        plt.show()
        
    def plot_2d_trajectories_ascii(self, results: Dict, model, title: str = "2D Trajectories"):
        """Plot 2D trajectories using ASCII characters"""
        
        if model.param_shape != (2,):
            console.print("❌ 2D plotting only works for 2-parameter models")
            return
            
        plt.clf()
        plt.title(title)
        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
        
        colors = ['red', 'blue', 'green', 'magenta']
        
        for i, (name, trajectory) in enumerate(results.items()):
            params_array = np.array(trajectory.params)
            x_vals = params_array[:, 0]
            y_vals = params_array[:, 1]
            
            plt.plot(x_vals, y_vals, label=name, color=colors[i % len(colors)])
            # Mark start and end points
            plt.plotsize(80, 25)  # Set reasonable plot size
        
        plt.show()
    
    def plot_3d_trajectories_ascii(self, results: Dict, title: str = "3D Trajectories"):
        """Plot 3D trajectories using first 3 parameters"""
        
        plt.clf()
        plt.title(title)
        
        colors = ['red', 'blue', 'green', 'magenta']
        
        for i, (name, trajectory) in enumerate(results.items()):
            params_array = np.array(trajectory.params)
            
            # Use first 3 dimensions
            x_vals = params_array[:, 0]
            y_vals = params_array[:, 1] 
            z_vals = params_array[:, 2] if params_array.shape[1] > 2 else params_array[:, 0] * 0  # Use zeros if only 2D
            
            plt.scatter3d(x_vals, y_vals, z_vals, label=name, color=colors[i % len(colors)])
        
        plt.plotsize(80, 30)  # Larger for 3D
        plt.show()
    
    def plot_3d_loss_surface(self, model, x_range=(-3, 3), y_range=(-3, 3), resolution=20):
        """Plot 3D loss surface for 2-parameter models"""
        
        if model.param_shape != (2,):
            console.print("❌ 3D surface plotting only works for 2-parameter models")
            return
        
        console.print("🎨 Generating 3D loss surface...")
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute loss surface
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                params = jnp.array([X[i, j], Y[i, j]])
                Z[i, j] = model.loss(params)
        
        plt.clf()
        plt.title("3D Loss Surface")
        
        # Flatten for 3D plotting
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        plt.scatter3d(x_flat, y_flat, z_flat, color='gray', marker='small')
        plt.plotsize(80, 30)
        plt.show()
    
    def create_results_table(self, results: Dict) -> Table:
        """Create a rich table with results"""
        
        analysis = analyze_convergence(results)
        
        table = Table(title="🏆 Optimizer Comparison Results")
        table.add_column("Optimizer", style="cyan", no_wrap=True)
        table.add_column("Final Loss", style="magenta", justify="right")
        table.add_column("Convergence Rate", style="green", justify="right")
        table.add_column("Steps to 90%", style="yellow", justify="right")
        table.add_column("Avg Step Size", style="blue", justify="right")
        
        # Sort by final loss
        sorted_results = sorted(analysis.items(), key=lambda x: x[1]['final_loss'])
        
        for name, metrics in sorted_results:
            table.add_row(
                name,
                f"{metrics['final_loss']:.2e}",
                f"{metrics['convergence_rate']:.1f}x",
                f"{metrics['convergence_step']}",
                f"{metrics['avg_step_size']:.2e}"
            )
        
        return table
    
    def run_comparison(self, model_name: str, optimizer_names: List[str], steps: int = 500):
        """Run optimizer comparison with terminal output"""
        
        # Setup model
        if model_name == 'rosenbrock':
            model = RosenbrockModel(dim=2)
            initial_params = jnp.array([-2.0, 2.0])
            description = "Non-convex function with narrow valley"
        elif model_name == 'rosenbrock3d':
            model = RosenbrockModel(dim=3)
            initial_params = jnp.array([-2.0, 2.0, 1.0])
            description = "3D Rosenbrock function"
        elif model_name == 'quadratic':
            model = SimpleQuadraticModel(dim=2, condition_number=10)
            initial_params = jnp.array([3.0, 4.0])
            description = "Well-conditioned quadratic bowl"
        elif model_name == 'quadratic3d':
            model = SimpleQuadraticModel(dim=3, condition_number=50)
            initial_params = jnp.array([3.0, 4.0, -2.0])
            description = "3D well-conditioned quadratic bowl"
        elif model_name == 'ill_conditioned':
            model = SimpleQuadraticModel(dim=10, condition_number=1000)
            initial_params = jax.random.normal(jax.random.PRNGKey(42), (10,)) * 2.0
            description = "High-dimensional ill-conditioned problem"
        else:
            console.print(f"❌ Unknown model: {model_name}")
            return
        
        # Select optimizers
        selected_optimizers = {name: self.optimizers[name] for name in optimizer_names}
        
        # Show problem info
        console.print(Panel(
            f"[bold blue]{model_name.title()}[/bold blue]\n"
            f"{description}\n"
            f"Dimensions: {model.param_shape[0]}\n"
            f"Initial params: {initial_params}\n"
            f"Optimizers: {', '.join(optimizer_names)}\n"
            f"Steps: {steps}",
            title="🎯 Problem Setup"
        ))
        
        # Run optimization with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Running optimization...", total=100)
            
            results = compare_optimizers(
                model=model,
                optimizer_configs=selected_optimizers,
                initial_params=initial_params,
                batches=[None],
                num_steps=steps,
                verbose=False
            )
            
            progress.update(task, completed=100)
        
        # Display results table
        console.print("\n")
        console.print(self.create_results_table(results))
        
        # Plot loss curves
        console.print("\n📈 Loss Curves:")
        self.plot_loss_curves(results, f"{model_name.title()} - Loss Evolution")
        
        # Plot trajectories
        if model.param_shape == (2,):
            console.print("\n🗺️  2D Trajectories:")
            self.plot_2d_trajectories_ascii(results, model, f"{model_name.title()} - Parameter Trajectories")
            
            # Also show 3D surface for 2D models
            console.print("\n🏔️  3D Loss Surface:")
            self.plot_3d_loss_surface(model)
        elif model.param_shape[0] >= 3:
            console.print("\n📊 3D Trajectory Projection (First 3 Parameters):")
            self.plot_3d_trajectories_ascii(results, f"{model_name.title()} - 3D Parameter Evolution")
        
        return results
    
    def interactive_menu(self):
        """Interactive menu system"""
        
        console.print(Panel(
            "[bold cyan]🚀 Terminal Optimizer Demo[/bold cyan]\n\n"
            "Compare optimizers directly in your terminal!\n"
            "All plots render as beautiful ASCII art.",
            title="Welcome"
        ))
        
        while True:
            console.print("\n" + "="*60)
            
            # Create options layout
            model_options = Panel(
                "[bold]Models:[/bold]\n"
                "1. rosenbrock (2D non-convex)\n"
                "2. rosenbrock3d (3D non-convex)\n"
                "3. quadratic (2D well-conditioned)\n"
                "4. quadratic3d (3D well-conditioned)\n"
                "5. ill_conditioned (10D, cond=1000)",
                title="📊 Models",
                width=35
            )
            
            optimizer_options = Panel(
                "[bold]Optimizers:[/bold]\n"
                "• Adam\n"
                "• AdamW\n" 
                "• SGD\n"
                "• Muon",
                title="⚡ Optimizers",
                width=30
            )
            
            console.print(Columns([model_options, optimizer_options]))
            
            console.print("\n[bold yellow]Commands:[/bold yellow]")
            console.print("  [cyan]run <model> <opt1,opt2,...> [steps][/cyan]  - Run comparison")
            console.print("  [cyan]quick[/cyan]                              - Quick demo")
            console.print("  [cyan]quit[/cyan]                               - Exit")
            
            cmd = console.input("\n[bold green]>[/bold green] ").strip().lower()
            
            if cmd == 'quit':
                console.print("👋 [bold blue]Thanks for using the terminal optimizer demo![/bold blue]")
                break
            elif cmd == 'quick':
                console.print("🚀 [yellow]Running quick demo...[/yellow]")
                self.run_comparison('rosenbrock', ['Adam', 'SGD'], steps=300)
            elif cmd.startswith('run'):
                self._handle_run_command(cmd)
            else:
                console.print("❌ [red]Unknown command[/red]")
    
    def _handle_run_command(self, cmd: str):
        """Handle run command"""
        try:
            parts = cmd.split()
            if len(parts) < 3:
                console.print("❌ [red]Usage:[/red] run <model> <optimizer1,optimizer2,...> [steps]")
                console.print("   [dim]Example:[/dim] run rosenbrock Adam,SGD,Muon 500")
                return
            
            model_name = parts[1]
            optimizer_names = [name.title() for name in parts[2].split(',')]
            steps = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 500
            
            # Validate
            valid_models = ['rosenbrock', 'rosenbrock3d', 'quadratic', 'quadratic3d', 'ill_conditioned']
            if model_name not in valid_models:
                console.print(f"❌ [red]Model must be one of:[/red] {valid_models}")
                return
            
            invalid_optimizers = [name for name in optimizer_names if name not in self.optimizers]
            if invalid_optimizers:
                console.print(f"❌ [red]Invalid optimizers:[/red] {invalid_optimizers}")
                console.print(f"   [dim]Valid:[/dim] {list(self.optimizers.keys())}")
                return
            
            # Run comparison
            self.run_comparison(model_name, optimizer_names, steps)
            
        except Exception as e:
            console.print(f"❌ [red]Error:[/red] {e}")


def main():
    """Main entry point"""
    
    if not TERMINAL_LIBS_AVAILABLE:
        return
    
    demo = TerminalOptimizer()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            console.print("🚀 [bold yellow]Quick terminal demo[/bold yellow]")
            demo.run_comparison('rosenbrock', ['Adam', 'SGD', 'Muon'], steps=300)
        else:
            console.print("❌ [red]Usage:[/red] python terminal_demo.py [quick]")
    else:
        # Interactive mode
        demo.interactive_menu()


if __name__ == "__main__":
    main()