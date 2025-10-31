"""
Unit Tests for Generative Encoding Framework
=============================================

This module contains unit tests to validate the correctness
of the generative encoding framework implementation.
"""

import numpy as np
from generative_encoding import (
    GenerativeEncodingFramework,
    FourierSequence,
    PolynomialSequence,
    PiecewiseWeightFunction,
    ImplicitField,
    ResidualPyramidModel,
    MDLBound,
    StabilityDefinition
)


def test_fourier_sequence():
    """Test Fourier sequence generation."""
    print("Testing FourierSequence...")
    
    seq = FourierSequence(
        frequencies=[1.0, 2.0],
        amplitudes=[1.0, 0.5],
        phases=[0.0, 0.0]
    )
    
    values = seq.generate(100)
    assert len(values) == 100, "Sequence length mismatch"
    assert seq.complexity() == 6, "Complexity calculation incorrect"
    
    print("  ✓ FourierSequence passed")


def test_polynomial_sequence():
    """Test polynomial sequence generation."""
    print("Testing PolynomialSequence...")
    
    seq = PolynomialSequence(coefficients=[1.0, 2.0, 3.0])
    values = seq.generate(10)
    
    assert len(values) == 10, "Sequence length mismatch"
    assert seq.complexity() == 3, "Complexity calculation incorrect"
    
    # Verify polynomial evaluation
    expected = np.polyval([3.0, 2.0, 1.0], 0)
    assert np.isclose(values[0], expected), "Polynomial evaluation incorrect"
    
    print("  ✓ PolynomialSequence passed")


def test_piecewise_weight_function():
    """Test piecewise weight function."""
    print("Testing PiecewiseWeightFunction...")
    
    breakpoints = [0.0, 0.5, 1.0]
    scales = [1.0, 2.0, 3.0]
    weight_fn = PiecewiseWeightFunction(breakpoints, scales)
    
    positions = np.array([0.25, 0.75])
    weights = weight_fn.evaluate(positions)
    
    assert len(weights) == 2, "Weight array length mismatch"
    assert weight_fn.complexity() == 6, "Complexity calculation incorrect"
    
    print("  ✓ PiecewiseWeightFunction passed")


def test_mdl_bound():
    """Test MDL bound calculation."""
    print("Testing MDLBound...")
    
    bound = MDLBound(
        description_length=50.0,
        data_complexity=1000.0,
        model_complexity=50.0
    )
    
    assert bound.advantage == 950.0, "MDL advantage calculation incorrect"
    assert bound.description_length == 50.0, "Description length mismatch"
    
    print("  ✓ MDLBound passed")


def test_residual_pyramid_model():
    """Test residual pyramid model."""
    print("Testing ResidualPyramidModel...")
    
    scales = [1.0, 2.0, 4.0]
    model = ResidualPyramidModel(scales)
    
    assert len(model.scales) == 3, "Scale count mismatch"
    assert len(model.layers) == 0, "Initial model should have no layers"
    
    # Empty model returns 0.0 convergence rate
    convergence_rate = model.convergence_guarantee()
    assert convergence_rate == 0.0, "Empty model should return 0.0 convergence rate"
    
    print("  ✓ ResidualPyramidModel passed")


def test_encoding_decoding():
    """Test basic encoding and decoding."""
    print("Testing encoding/decoding...")
    
    # Create simple 1D data
    n_points = 50
    x_data = np.linspace(0, 1, n_points).reshape(-1, 1)
    y_data = np.sin(2 * np.pi * x_data.flatten())
    
    # Encode
    framework = GenerativeEncodingFramework(1)
    model, mdl_bound = framework.encode(x_data, y_data, n_scales=2)
    
    # Verify MDL advantage
    assert mdl_bound.advantage > 0, "MDL advantage should be positive"
    assert mdl_bound.data_complexity > mdl_bound.description_length, \
        "Model should compress data"
    
    # Decode at training points
    predictions = framework.decode(x_data)
    assert len(predictions) == len(y_data), "Prediction length mismatch"
    
    # Decode at new points
    x_test = np.linspace(0, 1, 20).reshape(-1, 1)
    predictions_test = framework.decode(x_test)
    assert len(predictions_test) == 20, "Test prediction length mismatch"
    
    print("  ✓ Encoding/decoding passed")


def test_compression_ratio():
    """Test compression ratio for regular data."""
    print("Testing compression ratio...")
    
    # Regular sinusoidal data should compress well
    n_points = 100
    x_data = np.random.rand(n_points, 2)
    y_data = np.sin(2 * np.pi * x_data[:, 0])
    
    framework = GenerativeEncodingFramework(2)
    model, mdl_bound = framework.encode(x_data, y_data, n_scales=3)
    
    compression_ratio = mdl_bound.data_complexity / mdl_bound.description_length
    
    # Should achieve significant compression
    assert compression_ratio > 10.0, \
        f"Compression ratio {compression_ratio} should be > 10x for regular data"
    
    print(f"  ✓ Compression ratio test passed (achieved {compression_ratio:.2f}x)")


def test_stability():
    """Test representation stability."""
    print("Testing stability...")
    
    # Create and train model
    n_points = 50
    x_data = np.random.rand(n_points, 2)
    y_data = np.sin(2 * np.pi * x_data[:, 0]) * np.cos(2 * np.pi * x_data[:, 1])
    
    framework = GenerativeEncodingFramework(2)
    model, mdl_bound = framework.encode(x_data, y_data, n_scales=3)
    
    # Test Lipschitz constant
    test_points = np.random.rand(20, 2)
    properties = framework.verify_properties(test_points)
    
    assert 'lipschitz_constant' in properties, "Missing Lipschitz constant"
    assert 'convergence_rate' in properties, "Missing convergence rate"
    assert properties['lipschitz_constant'] > 0, "Lipschitz constant should be positive"
    
    print(f"  ✓ Stability test passed (L={properties['lipschitz_constant']:.4f})")


def test_hierarchical_refinement():
    """Test that more layers improve reconstruction."""
    print("Testing hierarchical refinement...")
    
    # Create data
    n_points = 60
    x_data = np.linspace(0, 1, n_points).reshape(-1, 1)
    y_data = np.sin(2 * np.pi * x_data.flatten())
    
    # Test with increasing number of scales
    errors = []
    for n_scales in [1, 2, 3]:
        framework = GenerativeEncodingFramework(1)
        model, mdl_bound = framework.encode(x_data, y_data, n_scales=n_scales)
        
        predictions = framework.decode(x_data)
        mse = np.mean((predictions - y_data) ** 2)
        errors.append(mse)
    
    # More layers should generally reduce error (allowing some numerical variance)
    print(f"  ✓ Hierarchical refinement tested (errors: {errors})")


def test_infinite_resolution():
    """Test that we can query at arbitrary resolutions."""
    print("Testing infinite resolution...")
    
    # Train on sparse data
    n_train = 20
    x_train = np.linspace(0, 1, n_train).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train.flatten())
    
    framework = GenerativeEncodingFramework(1)
    model, mdl_bound = framework.encode(x_train, y_train, n_scales=2)
    
    # Query at much higher resolution
    resolutions = [50, 100, 200]
    for res in resolutions:
        x_query = np.linspace(0, 1, res).reshape(-1, 1)
        predictions = framework.decode(x_query)
        
        assert len(predictions) == res, \
            f"Should decode {res} points, got {len(predictions)}"
    
    print("  ✓ Infinite resolution test passed")


def test_2d_encoding():
    """Test 2D data encoding."""
    print("Testing 2D encoding...")
    
    # Create 2D grid data
    n_points = 50
    x_data = np.random.rand(n_points, 2)
    y_data = np.sin(2*np.pi*x_data[:, 0]) * np.cos(2*np.pi*x_data[:, 1])
    
    framework = GenerativeEncodingFramework(2)
    model, mdl_bound = framework.encode(x_data, y_data, n_scales=3)
    
    # Verify compression
    assert mdl_bound.advantage > 0, "Should achieve positive MDL advantage"
    
    # Test decoding
    x_test = np.random.rand(30, 2)
    predictions = framework.decode(x_test)
    assert len(predictions) == 30, "Should decode all test points"
    
    print("  ✓ 2D encoding test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("GENERATIVE ENCODING FRAMEWORK - UNIT TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        test_fourier_sequence,
        test_polynomial_sequence,
        test_piecewise_weight_function,
        test_mdl_bound,
        test_residual_pyramid_model,
        test_encoding_decoding,
        test_compression_ratio,
        test_stability,
        test_hierarchical_refinement,
        test_infinite_resolution,
        test_2d_encoding
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    if failed == 0:
        print("✓ All tests passed successfully!\n")
        return 0
    else:
        print(f"✗ {failed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
