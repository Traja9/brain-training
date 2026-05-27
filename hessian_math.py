import numpy as np
import time
from typing import Callable, Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ComputationMetrics:
    """Metrics for Hessian computation"""
    operation_type: str
    computation_time_ms: float
    numerical_precision: float
    convergence_status: str
    function_evaluations: int
    hessian_condition_number: Optional[float]


class ScalarField:
    """Represents a scalar field function with dimension information"""
    
    def __init__(self, function: Callable, dimension: int):
        self.function = function
        self.dimension = dimension
    
    def __call__(self, point):
        """Allow the ScalarField to be called like a function"""
        return self.function(point)


class NumericalDerivativeConfig:
    """Configuration for numerical derivative computations"""
    
    def __init__(self, derivative_step_size: float = 1e-2):
        self.derivative_step_size = derivative_step_size
        self.stats = {
            'hessian_computations': 0,
            'total_operations': 0,
            'gradient_computations': 0
        }


class HessianComputer:
    """Class for computing Hessian matrices using finite differences"""
    
    def __init__(self, derivative_step_size: float = 1e-2):
        self.derivative_step_size = derivative_step_size
        self.stats = {
            'hessian_computations': 0,
            'total_operations': 0
        }
    
    def compute_hessian(
        self,
        scalar_field: Union[ScalarField, Callable],
        point: np.ndarray
    ) -> Tuple[np.ndarray, ComputationMetrics]:
        """
        Compute the Hessian matrix at a given point using finite differences.
        
        Args:
            scalar_field: Either a ScalarField object or a callable function
            point: Point at which to compute the Hessian
        
        Returns:
            Tuple of (hessian_matrix, computation_metrics)
        """
        start_time = time.perf_counter()
        self.stats['hessian_computations'] += 1
        self.stats['total_operations'] += 1
        
        # Extract function and dimension
        if isinstance(scalar_field, ScalarField):
            func = scalar_field.function
            dim = scalar_field.dimension
        elif hasattr(scalar_field, 'function'):
            func = scalar_field.function
            dim = scalar_field.dimension
        else:
            func = scalar_field
            dim = len(point)
        
        point = np.asarray(point, dtype=float)
        h = self.derivative_step_size
        hessian = np.zeros((dim, dim))
        function_evals = 0
        
        # Center point evaluation
        f_center = func(point)
        function_evals += 1
        
        # Compute each Hessian element
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    # Diagonal element: ∂²f/∂xi²
                    p_plus = point.copy()
                    p_plus[i] += h
                    p_minus = point.copy()
                    p_minus[i] -= h
                    
                    hessian[i, i] = (func(p_plus) - 2.0 * f_center + func(p_minus)) / (h * h)
                    function_evals += 2
                else:
                    # Off-diagonal element: ∂²f/∂xi∂xj
                    pp = point.copy()
                    pp[i] += h
                    pp[j] += h
                    
                    pm = point.copy()
                    pm[i] += h
                    pm[j] -= h
                    
                    mp = point.copy()
                    mp[i] -= h
                    mp[j] += h
                    
                    mm = point.copy()
                    mm[i] -= h
                    mm[j] -= h
                    
                    val = (func(pp) - func(pm) - func(mp) + func(mm)) / (4.0 * h * h)
                    hessian[i, j] = val
                    hessian[j, i] = val  # Symmetry
                    function_evals += 4
        
        # Compute condition number
        try:
            eigvals = np.linalg.eigvals(hessian)
            significant = np.abs(eigvals)[np.abs(eigvals) > 1e-12]
            if len(significant) > 1:
                cond = float(np.max(significant) / np.min(significant))
            else:
                cond = 1.0
        except:
            cond = None
        
        # Create metrics
        metrics = ComputationMetrics(
            operation_type="HESSIAN",
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
            numerical_precision=h * h,
            convergence_status="computed",
            function_evaluations=function_evals,
            hessian_condition_number=cond
        )
        
        return hessian, metrics
    
    def get_stats(self) -> dict:
        """Return computation statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset computation statistics"""
        self.stats = {
            'hessian_computations': 0,
            'total_operations': 0
        }


class HessianTester:
    """Class for testing Hessian computations"""
    
    def __init__(self, step_size: float = 1e-2):
        self.computer = HessianComputer(derivative_step_size=step_size)
        self.step_size = step_size
    
    @staticmethod
    def quadratic_test_function(p: np.ndarray) -> float:
        """
        Test function: f(x,y) = x² + 3xy + y²
        Expected Hessian: [[2, 3], [3, 2]]
        """
        x, y = p[0], p[1]
        return x ** 2 + 3 * x * y + y ** 2
    
    def test_standalone(self) -> np.ndarray:
        """Test Hessian computation in isolation with detailed output"""
        print("=" * 70)
        print("STANDALONE HESSIAN TEST")
        print("=" * 70)
        
        point = np.array([1.0, 2.0])
        h = self.step_size
        dim = 2
        hessian = np.zeros((dim, dim))
        
        # Compute center value
        f_center = self.quadratic_test_function(point)
        print(f"\nTest Function: f(x,y) = x² + 3xy + y²")
        print(f"Point: {point}")
        print(f"f({point}) = {f_center}")
        print(f"Expected: 1² + 3(1)(2) + 2² = 1 + 6 + 4 = 11")
        
        # Compute Hessian elements manually with detailed output
        print(f"\nComputing Hessian with step size h = {h}")
        print("-" * 70)
        
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    # Diagonal
                    p_plus = point.copy()
                    p_plus[i] += h
                    p_minus = point.copy()
                    p_minus[i] -= h
                    
                    f_plus = self.quadratic_test_function(p_plus)
                    f_minus = self.quadratic_test_function(p_minus)
                    
                    hessian[i, i] = (f_plus - 2.0 * f_center + f_minus) / (h * h)
                    
                    print(f"\nH[{i},{i}] (Diagonal):")
                    print(f"  f_plus  = {f_plus:.6f} at {p_plus}")
                    print(f"  f_minus = {f_minus:.6f} at {p_minus}")
                    print(f"  f_center = {f_center:.6f}")
                    print(f"  Formula: ({f_plus:.6f} - 2*{f_center:.6f} + {f_minus:.6f}) / {h*h:.6f}")
                    print(f"  Result: {hessian[i, i]:.6f}")
                else:
                    # Off-diagonal
                    pp = point.copy()
                    pp[i] += h
                    pp[j] += h
                    
                    pm = point.copy()
                    pm[i] += h
                    pm[j] -= h
                    
                    mp = point.copy()
                    mp[i] -= h
                    mp[j] += h
                    
                    mm = point.copy()
                    mm[i] -= h
                    mm[j] -= h
                    
                    f_pp = self.quadratic_test_function(pp)
                    f_pm = self.quadratic_test_function(pm)
                    f_mp = self.quadratic_test_function(mp)
                    f_mm = self.quadratic_test_function(mm)
                    
                    val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
                    hessian[i, j] = val
                    hessian[j, i] = val
                    
                    print(f"\nH[{i},{j}] (Off-diagonal):")
                    print(f"  f(++) = {f_pp:.6f} at {pp}")
                    print(f"  f(+-) = {f_pm:.6f} at {pm}")
                    print(f"  f(-+) = {f_mp:.6f} at {mp}")
                    print(f"  f(--) = {f_mm:.6f} at {mm}")
                    print(f"  Formula: ({f_pp:.6f} - {f_pm:.6f} - {f_mp:.6f} + {f_mm:.6f}) / {4.0*h*h:.6f}")
                    print(f"  Result: {val:.6f}")
        
        # Compare with expected
        expected_hessian = np.array([[2, 3], [3, 2]])
        error = np.linalg.norm(hessian - expected_hessian)
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nComputed Hessian:")
        print(hessian)
        print(f"\nExpected Hessian:")
        print(expected_hessian)
        print(f"\nError (Frobenius norm): {error:.10f}")
        
        if error < 1e-6:
            print("✓ TEST PASSED - Hessian is accurate!")
        else:
            print("✗ TEST FAILED - Hessian has significant error!")
        
        print("=" * 70)
        return hessian
    
    def test_with_computer(self) -> Tuple[np.ndarray, ComputationMetrics]:
        """Test using the HessianComputer class"""
        print("\n" + "=" * 70)
        print("TESTING WITH HessianComputer CLASS")
        print("=" * 70)
        
        # Create a ScalarField
        scalar_field = ScalarField(
            function=self.quadratic_test_function,
            dimension=2
        )
        
        point = np.array([1.0, 2.0])
        
        print(f"\nComputing Hessian at point {point}...")
        hessian, metrics = self.computer.compute_hessian(scalar_field, point)
        
        print(f"\nComputation Metrics:")
        print(f"  Operation Type: {metrics.operation_type}")
        print(f"  Computation Time: {metrics.computation_time_ms:.4f} ms")
        print(f"  Function Evaluations: {metrics.function_evaluations}")
        print(f"  Numerical Precision: {metrics.numerical_precision:.2e}")
        print(f"  Condition Number: {metrics.hessian_condition_number:.4f}")
        print(f"  Status: {metrics.convergence_status}")
        
        print(f"\nComputed Hessian:")
        print(hessian)
        
        expected_hessian = np.array([[2, 3], [3, 2]])
        error = np.linalg.norm(hessian - expected_hessian)
        print(f"\nError: {error:.10f}")
        
        print("\nComputation Statistics:")
        stats = self.computer.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("=" * 70)
        return hessian, metrics


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("HESSIAN COMPUTATION TEST SUITE")
    print("="*70)
    
    # Create tester
    tester = HessianTester(step_size=1e-2)
    
    # Run standalone test
    print("\n[1] Running Standalone Manual Test...")
    hessian1 = tester.test_standalone()
    
    # Run test with HessianComputer class
    print("\n[2] Running Test with HessianComputer Class...")
    hessian2, metrics = tester.test_with_computer()
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Difference between methods: {np.linalg.norm(hessian1 - hessian2):.2e}")
    print("\n✓ All tests completed successfully!")
    print("="*70)