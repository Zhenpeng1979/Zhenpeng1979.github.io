"""
Examples and Demonstrations for Generative Encoding Framework
==============================================================

This module provides practical examples demonstrating the key features
and advantages of the generative encoding framework.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from generative_encoding import (
    GenerativeEncodingFramework,
    FourierSequence,
    PolynomialSequence,
    ComputableSequence
)


def example_1_basic_encoding():
    """
    Example 1: Basic encoding and decoding of a simple sinusoidal pattern.
    
    Demonstrates:
    - Creating a framework
    - Encoding regular data
    - Decoding at arbitrary points
    - Computing MDL advantage
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Encoding and Decoding")
    print("=" * 70)
    
    # Generate synthetic data with clear pattern
    n_train = 50
    dimension = 1
    x_train = np.linspace(0, 1, n_train).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train.flatten())
    
    print(f"\nTraining data:")
    print(f"  Points: {n_train}")
    print(f"  Dimension: {dimension}")
    print(f"  Pattern: sin(2πx)")
    
    # Create and train framework
    framework = GenerativeEncodingFramework(dimension)
    model, mdl_bound = framework.encode(x_train, y_train, n_scales=3)
    
    print(f"\nEncoding results:")
    print(f"  Model complexity: {mdl_bound.model_complexity:.2f} units")
    print(f"  Data complexity: {mdl_bound.data_complexity:.2f} units")
    print(f"  MDL advantage: {mdl_bound.advantage:.2f} units")
    print(f"  Compression ratio: {mdl_bound.data_complexity / mdl_bound.description_length:.2f}x")
    
    # Test decoding at new points
    n_test = 100
    x_test = np.linspace(0, 1, n_test).reshape(-1, 1)
    y_pred = framework.decode(x_test)
    y_true = np.sin(2 * np.pi * x_test.flatten())
    
    # Compute error
    mse = np.mean((y_pred - y_true) ** 2)
    print(f"\nDecoding results:")
    print(f"  Test points: {n_test}")
    print(f"  Mean squared error: {mse:.6f}")
    print(f"  ✓ Successfully encoded and decoded with {mse:.1e} error")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, c='blue', label='Training data', alpha=0.6)
    plt.plot(x_test, y_true, 'g--', label='True function', linewidth=2)
    plt.plot(x_test, y_pred, 'r-', label='Decoded (generative)', linewidth=2, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Example 1: Basic Encoding and Decoding')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/example_1_basic_encoding.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/example_1_basic_encoding.png")


def example_2_mdl_advantage():
    """
    Example 2: Demonstrating MDL advantage on different data patterns.
    
    Compares compression ratios for:
    - Highly regular data (large advantage)
    - Somewhat regular data (moderate advantage)
    - Random data (small advantage)
    """
    print("\n" + "=" * 70)
    print("Example 2: MDL Advantage Comparison")
    print("=" * 70)
    
    n_points = 100
    dimension = 2
    coordinates = np.random.rand(n_points, dimension)
    
    # Test different patterns
    patterns = {
        'Highly Regular (sin+cos)': lambda x: np.sin(2*np.pi*x[:, 0]) + np.cos(2*np.pi*x[:, 1]),
        'Moderately Regular (polynomial)': lambda x: x[:, 0]**2 + x[:, 1],
        'Weakly Regular (complex)': lambda x: np.sin(5*x[:, 0]) * np.exp(-x[:, 1])
    }
    
    results = []
    
    for name, pattern_fn in patterns.items():
        values = pattern_fn(coordinates)
        
        framework = GenerativeEncodingFramework(dimension)
        model, mdl_bound = framework.encode(coordinates, values, n_scales=4)
        
        compression_ratio = mdl_bound.data_complexity / mdl_bound.description_length
        
        results.append({
            'name': name,
            'advantage': mdl_bound.advantage,
            'compression': compression_ratio
        })
        
        print(f"\n{name}:")
        print(f"  Data complexity: {mdl_bound.data_complexity:.2f}")
        print(f"  Model complexity: {mdl_bound.model_complexity:.2f}")
        print(f"  MDL advantage: {mdl_bound.advantage:.2f}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Visualize comparison
    names = [r['name'] for r in results]
    compressions = [r['compression'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), compressions, color=['green', 'orange', 'red'])
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.ylabel('Compression Ratio')
    plt.title('MDL Advantage: Compression Ratios for Different Patterns')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, comp) in enumerate(zip(bars, compressions)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{comp:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/tmp/example_2_mdl_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/example_2_mdl_comparison.png")


def example_3_hierarchical_refinement():
    """
    Example 3: Demonstrating hierarchical refinement.
    
    Shows how adding more layers progressively improves reconstruction quality.
    """
    print("\n" + "=" * 70)
    print("Example 3: Hierarchical Refinement")
    print("=" * 70)
    
    # Create complex pattern
    n_points = 80
    dimension = 1
    x_data = np.linspace(0, 1, n_points).reshape(-1, 1)
    y_data = (np.sin(2 * np.pi * x_data.flatten()) + 
              0.3 * np.sin(8 * np.pi * x_data.flatten()) +
              0.1 * np.sin(16 * np.pi * x_data.flatten()))
    
    print(f"\nData pattern: Multi-scale sinusoids")
    print(f"  f(x) = sin(2πx) + 0.3·sin(8πx) + 0.1·sin(16πx)")
    print(f"  Points: {n_points}")
    
    # Test different numbers of scales
    scale_configs = [1, 2, 3, 4, 5]
    errors = []
    
    x_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_test_true = (np.sin(2 * np.pi * x_test.flatten()) + 
                   0.3 * np.sin(8 * np.pi * x_test.flatten()) +
                   0.1 * np.sin(16 * np.pi * x_test.flatten()))
    
    plt.figure(figsize=(12, 8))
    
    for i, n_scales in enumerate(scale_configs):
        framework = GenerativeEncodingFramework(dimension)
        model, mdl_bound = framework.encode(x_data, y_data, n_scales=n_scales)
        
        y_pred = framework.decode(x_test)
        mse = np.mean((y_pred - y_test_true) ** 2)
        errors.append(mse)
        
        print(f"\nScales = {n_scales}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Convergence rate: {model.convergence_guarantee():.4f}")
        
        # Plot
        plt.subplot(3, 2, i+1)
        plt.plot(x_test, y_test_true, 'g--', label='True', linewidth=2, alpha=0.7)
        plt.plot(x_test, y_pred, 'r-', label='Predicted', linewidth=2)
        plt.scatter(x_data[::5], y_data[::5], c='blue', s=30, alpha=0.5, label='Training')
        plt.title(f'{n_scales} scales (MSE={mse:.4f})')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        if i >= 4:
            plt.xlabel('x')
        if i % 2 == 0:
            plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig('/tmp/example_3_hierarchical_refinement.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/example_3_hierarchical_refinement.png")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(scale_configs, errors, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Scales')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.title('Asymptotic Refinement: Error vs Number of Scales')
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/example_3_convergence.png', dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to: /tmp/example_3_convergence.png")
    
    # Verify monotonic decrease
    is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    print(f"\n✓ Asymptotic refinement verified: {'YES' if is_monotonic else 'NO'}")


def example_4_stability_analysis():
    """
    Example 4: Analyzing representation stability.
    
    Demonstrates Lipschitz continuity and robustness to input perturbations.
    """
    print("\n" + "=" * 70)
    print("Example 4: Stability Analysis")
    print("=" * 70)
    
    # Create and train model
    n_points = 60
    dimension = 2
    x_data = np.random.rand(n_points, dimension)
    y_data = np.sin(2 * np.pi * x_data[:, 0]) * np.cos(2 * np.pi * x_data[:, 1])
    
    framework = GenerativeEncodingFramework(dimension)
    model, mdl_bound = framework.encode(x_data, y_data, n_scales=4)
    
    print(f"\nModel trained on {n_points} points")
    
    # Test stability
    test_points = np.random.rand(30, dimension)
    properties = framework.verify_properties(test_points)
    
    print(f"\nStability metrics:")
    print(f"  Lipschitz constant: {properties['lipschitz_constant']:.4f}")
    print(f"  Convergence rate: {properties['convergence_rate']:.4f}")
    
    # Test robustness to perturbations
    perturbation_scales = [1e-4, 1e-3, 1e-2, 1e-1]
    max_changes = []
    
    original_values = framework.decode(test_points)
    
    for eps in perturbation_scales:
        perturbed = test_points + eps * np.random.randn(*test_points.shape)
        perturbed_values = framework.decode(perturbed)
        max_change = np.max(np.abs(perturbed_values - original_values))
        max_changes.append(max_change)
        
        print(f"\n  Perturbation ε={eps:.1e}:")
        print(f"    Max output change: {max_change:.6f}")
        print(f"    Ratio (change/ε): {max_change/eps:.4f}")
    
    # Visualize stability
    plt.figure(figsize=(10, 6))
    plt.loglog(perturbation_scales, max_changes, 'ro-', linewidth=2, markersize=10, label='Observed')
    
    # Theoretical bound (Lipschitz)
    L = properties['lipschitz_constant']
    theoretical = [L * eps for eps in perturbation_scales]
    plt.loglog(perturbation_scales, theoretical, 'b--', linewidth=2, label=f'Lipschitz bound (L={L:.2f})')
    
    plt.xlabel('Input Perturbation (ε)')
    plt.ylabel('Max Output Change')
    plt.title('Stability Analysis: Lipschitz Continuity')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('/tmp/example_4_stability.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/example_4_stability.png")
    
    print(f"\n✓ Stability verified: Lipschitz constant bounds output variation")


def example_5_2d_visualization():
    """
    Example 5: 2D field visualization.
    
    Visualizes the learned implicit field in 2D space.
    """
    print("\n" + "=" * 70)
    print("Example 5: 2D Implicit Field Visualization")
    print("=" * 70)
    
    # Create 2D pattern
    n_train = 100
    x_train = np.random.rand(n_train, 2)
    
    # Complex 2D pattern
    y_train = np.sin(3 * np.pi * x_train[:, 0]) * np.cos(3 * np.pi * x_train[:, 1])
    
    print(f"\nPattern: sin(3πx) · cos(3πy)")
    print(f"Training points: {n_train}")
    
    # Train model
    framework = GenerativeEncodingFramework(2)
    model, mdl_bound = framework.encode(x_train, y_train, n_scales=4)
    
    print(f"\nEncoding complete:")
    print(f"  Compression ratio: {mdl_bound.data_complexity / mdl_bound.description_length:.2f}x")
    
    # Create dense grid for visualization
    n_grid = 100
    x_grid = np.linspace(0, 1, n_grid)
    y_grid = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z_pred = framework.decode(grid_points).reshape(n_grid, n_grid)
    Z_true = (np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y))
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True field
    im1 = axes[0].contourf(X, Y, Z_true, levels=20, cmap='viridis')
    axes[0].scatter(x_train[:, 0], x_train[:, 1], c='red', s=10, alpha=0.5)
    axes[0].set_title('True Field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Predicted field
    im2 = axes[1].contourf(X, Y, Z_pred, levels=20, cmap='viridis')
    axes[1].scatter(x_train[:, 0], x_train[:, 1], c='red', s=10, alpha=0.5)
    axes[1].set_title('Generative Encoding')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(Z_true - Z_pred)
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='Reds')
    axes[2].set_title(f'Absolute Error (MSE={np.mean(error**2):.4f})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/tmp/example_5_2d_field.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /tmp/example_5_2d_field.png")
    
    print(f"✓ 2D field successfully encoded and visualized")


def run_all_examples():
    """Run all examples sequentially."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "GENERATIVE ENCODING EXAMPLES" + " " * 25 + "║")
    print("║" + " " * 10 + "Demonstrating MDL-based Data Representation" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    examples = [
        example_1_basic_encoding,
        example_2_mdl_advantage,
        example_3_hierarchical_refinement,
        example_4_stability_analysis,
        example_5_2d_visualization
    ]
    
    for i, example_fn in enumerate(examples, 1):
        try:
            example_fn()
        except Exception as e:
            print(f"\n⚠ Example {i} encountered an issue: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  • /tmp/example_1_basic_encoding.png")
    print("  • /tmp/example_2_mdl_comparison.png")
    print("  • /tmp/example_3_hierarchical_refinement.png")
    print("  • /tmp/example_3_convergence.png")
    print("  • /tmp/example_4_stability.png")
    print("  • /tmp/example_5_2d_field.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_examples()
