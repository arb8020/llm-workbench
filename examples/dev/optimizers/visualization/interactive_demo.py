#!/usr/bin/env python3
"""
Interactive Optimizer Demo

Clean, interactive command-line demo with minimal output and save-to-file options.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
from visualize import plot_2d_trajectories, plot_high_dimensional_trajectories


class InteractiveDemo:
    def __init__(self):
        self.optimizers = {
            'adam': OptimizerConfig('Adam', adam_init, adam_update, {'lr': 0.01}, 'red'),
            'adamw': OptimizerConfig('AdamW', adamw_init, adamw_update, {'lr': 0.01, 'weight_decay': 0.001}, 'blue'),
            'sgd': OptimizerConfig('SGD+Mom', sgd_init, sgd_update, {'lr': 0.002, 'momentum': 0.9}, 'green'),
            'muon': OptimizerConfig('Muon', muon_init, muon_update, {'lr': 0.005, 'momentum': 0.9}, 'purple')
        }
        
        self.models = {
            'rosenbrock': RosenbrockModel(dim=2),
            'quadratic': SimpleQuadraticModel(dim=2, condition_number=10),
            'ill_conditioned': SimpleQuadraticModel(dim=10, condition_number=1000)
        }
        
        self.results = {}
    
    def run_comparison(self, model_name: str, optimizer_names: list, steps: int = 500, save_plots: bool = False):
        """Run comparison and optionally save plots"""
        
        print(f"🚀 Running {model_name} with {optimizer_names} ({steps} steps)...")
        
        # Setup
        model = self.models[model_name]
        if model_name == 'rosenbrock':
            initial_params = jnp.array([-2.0, 2.0])
        elif model_name in ['quadratic', 'ill_conditioned']:
            key = jax.random.PRNGKey(42)
            initial_params = jax.random.normal(key, model.param_shape) * 1.0
        
        # Select optimizers
        selected_optimizers = {name: self.optimizers[name] for name in optimizer_names}
        
        # Run optimization
        results = compare_optimizers(
            model=model,
            optimizer_configs=selected_optimizers,
            initial_params=initial_params,
            batches=[None],
            num_steps=steps,
            verbose=False
        )
        
        # Store results
        self.results[f"{model_name}_{'-'.join(optimizer_names)}"] = {
            'results': results,
            'model': model,
            'model_name': model_name
        }
        
        # Quick summary
        analysis = analyze_convergence(results)
        print("Results:")
        for name, metrics in analysis.items():
            print(f"  {name:12} Final Loss: {metrics['final_loss']:.2e}  Steps to 90%: {metrics['convergence_step']:3d}")
        
        # Create plots if requested
        if save_plots:
            self._save_plots(f"{model_name}_{'-'.join(optimizer_names)}", results, model, model_name)
        
        return results
    
    def _save_plots(self, key: str, results, model, model_name: str):
        """Save plots to files"""
        
        output_dir = "optimizer_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2D trajectory plot (if 2D model)
        if model.param_shape == (2,):
            fig = plot_2d_trajectories(results, model, figsize=(10, 8))
            fig.suptitle(f"{model_name.title()} - Optimizer Trajectories")
            fig.savefig(f"{output_dir}/{key}_trajectories.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  💾 Saved: {output_dir}/{key}_trajectories.png")
        
        # High-dimensional projection (if high-dim)
        elif model.param_shape[0] > 2:
            fig = plot_high_dimensional_trajectories(results, method='pca', n_components=2)
            fig.suptitle(f"{model_name.title()} - PCA Projection")
            fig.savefig(f"{output_dir}/{key}_pca.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  💾 Saved: {output_dir}/{key}_pca.png")
    
    def show_menu(self):
        """Interactive menu"""
        while True:
            print("\n" + "="*50)
            print("🎯 Optimizer Comparison Demo")
            print("="*50)
            print("Models:")
            print("  1. Rosenbrock (2D, non-convex)")
            print("  2. Quadratic (2D, well-conditioned)")  
            print("  3. Ill-conditioned (10D, condition=1000)")
            print("\nOptimizers:")
            print("  adam, adamw, sgd, muon")
            print("\nCommands:")
            print("  run <model> <optimizers> [steps] [save]")
            print("  show <experiment_key>")
            print("  list")
            print("  quit")
            
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'list':
                print("Stored experiments:")
                for key in self.results.keys():
                    print(f"  {key}")
            elif cmd.startswith('run'):
                self._handle_run_command(cmd)
            elif cmd.startswith('show'):
                self._handle_show_command(cmd)
            else:
                print("❌ Unknown command")
    
    def _handle_run_command(self, cmd: str):
        """Handle run command"""
        try:
            parts = cmd.split()
            if len(parts) < 3:
                print("❌ Usage: run <model> <optimizer1,optimizer2,...> [steps] [save]")
                return
            
            model_name = parts[1]
            optimizer_names = parts[2].split(',')
            steps = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 500
            save_plots = len(parts) > 4 and parts[4].lower() in ['save', 'true', 'yes']
            
            # Validate inputs
            if model_name not in self.models:
                print(f"❌ Model must be one of: {list(self.models.keys())}")
                return
            
            invalid_optimizers = [name for name in optimizer_names if name not in self.optimizers]
            if invalid_optimizers:
                print(f"❌ Invalid optimizers: {invalid_optimizers}")
                return
            
            # Run comparison
            self.run_comparison(model_name, optimizer_names, steps, save_plots)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _handle_show_command(self, cmd: str):
        """Handle show command"""
        parts = cmd.split()
        if len(parts) != 2:
            print("❌ Usage: show <experiment_key>")
            return
        
        key = parts[1]
        if key not in self.results:
            print(f"❌ No experiment '{key}'. Use 'list' to see available experiments.")
            return
        
        # Show interactive plots
        experiment = self.results[key]
        results = experiment['results']
        model = experiment['model']
        model_name = experiment['model_name']
        
        if model.param_shape == (2,):
            fig = plot_2d_trajectories(results, model, figsize=(12, 9))
            fig.suptitle(f"{model_name.title()} - Optimizer Trajectories")
            plt.show()
        else:
            fig = plot_high_dimensional_trajectories(results, method='pca', n_components=2)
            fig.suptitle(f"{model_name.title()} - PCA Projection")
            plt.show()


def main():
    """Main demo function"""
    print("🎯 Interactive Optimizer Demo")
    print("This demo lets you compare optimizers interactively with clean output.")
    print("Type commands at the prompt, or 'quit' to exit.")
    
    demo = InteractiveDemo()
    
    # Quick demo run
    print("\n🚀 Quick Demo: Adam vs SGD on Rosenbrock")
    demo.run_comparison('rosenbrock', ['adam', 'sgd'], steps=300, save_plots=False)
    
    # Enter interactive mode
    demo.show_menu()
    
    print("👋 Thanks for using the optimizer demo!")


if __name__ == "__main__":
    main()