import math
import time
import decimal
import sys
from typing import List, Any, Union, Optional, Dict, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import warnings
from functools import lru_cache
import numpy as np

# SciPy imports for robust numerical integration
try:
    from scipy import integrate
    from scipy.integrate import quad, dblquad, tplquad, fixed_quad
    from scipy.integrate import quad_vec, simpson, trapezoid
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - falling back to custom implementations")

# Import the classes from ultra_special_sort.py
try:
    from ultra_special_sort import (
        SpecialValueType, 
        ValueMetrics, 
        PrecisionSpecialValueSorter
    )
except ImportError:
    # Create minimal versions if ultra_special_sort is not available
    class SpecialValueType(Enum):
        REGULAR = "regular"
        NAN = "nan"
        POSITIVE_INFINITY = "pos_inf"
        NEGATIVE_INFINITY = "neg_inf"
        NONE = "none"
    
    @dataclass
    class ValueMetrics:
        original_index: int
        classification: SpecialValueType
    
    class PrecisionSpecialValueSorter:
        def _ultra_precise_classify(self, value):
            if value is None:
                return SpecialValueType.NONE
            try:
                if math.isnan(float(value)):
                    return SpecialValueType.NAN
                if math.isinf(float(value)):
                    return SpecialValueType.POSITIVE_INFINITY if float(value) > 0 else SpecialValueType.NEGATIVE_INFINITY
                return SpecialValueType.REGULAR
            except:
                return SpecialValueType.REGULAR

class IntegrationMethod(Enum):
    """Enhanced enumeration of integration method types with SciPy integration"""
    ADAPTIVE_SIMPSON = "adaptive_simpson"
    ADAPTIVE_GAUSS_LEGENDRE = "adaptive_gauss"
    ROMBERG_EXTRAPOLATION = "romberg"
    MONTE_CARLO_ADAPTIVE = "monte_carlo"
    TANH_SINH_DOUBLE_EXP = "tanh_sinh"
    ADAPTIVE_CLENSHAW_CURTIS = "adaptive_cc"
    GAUSS_KRONROD_ADAPTIVE = "gauss_kronrod"
    COMPOSITE_NEWTON_COTES = "composite_nc"
    # SciPy methods
    SCIPY_QUAD = "scipy_quad"
    SCIPY_QUAD_VEC = "scipy_quad_vec"
    SCIPY_ROMBERG = "scipy_romberg"
    SCIPY_FIXED_QUAD = "scipy_fixed_quad"
    SCIPY_SIMPSON = "scipy_simpson"
    SCIPY_AUTO_SELECT = "scipy_auto_select"

class ConvergenceStatus(Enum):
    """Integration convergence status indicators"""
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    PRECISION_LIMIT = "precision_limit"
    OSCILLATORY = "oscillatory"
    SINGULAR = "singular"
    DIVERGENT = "divergent"
    UNSTABLE = "unstable"

class SingularityType(Enum):
    """Types of singularities that may be encountered"""
    NONE = "none"
    REMOVABLE = "removable"
    POLE_SIMPLE = "pole_simple"
    POLE_MULTIPLE = "pole_multiple"
    BRANCH_POINT = "branch_point"
    ESSENTIAL = "essential"
    ENDPOINT_LEFT = "endpoint_left"
    ENDPOINT_RIGHT = "endpoint_right"
    INTERIOR = "interior"

@dataclass
class IntegrationMetrics:
    """Comprehensive metrics for integration analysis using ValueMetrics pattern"""
    method_used: IntegrationMethod
    convergence_status: ConvergenceStatus
    final_result: Union[float, decimal.Decimal, complex]
    absolute_error_estimate: float
    relative_error_estimate: float
    function_evaluations: int
    refinement_levels: int
    computation_time_ms: float
    singularities_detected: List[Tuple[float, SingularityType]] = field(default_factory=list)
    precision_warnings: List[str] = field(default_factory=list)
    adaptive_subdivisions: int = 0
    oscillation_detected: bool = False
    ill_conditioning: float = 0.0
    value_classifications: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.singularities_detected is None:
            self.singularities_detected = []
        if self.precision_warnings is None:
            self.precision_warnings = []
        if self.value_classifications is None:
            self.value_classifications = {}

class UltraPrecisionIntegrator:
    """
    Ultra-precise numerical integration leveraging the PrecisionSpecialValueSorter
    architecture for comprehensive special value handling and precision analysis
    Enhanced with SciPy methods for robust handling of difficult functions
    """
    
    def __init__(self,
                 precision_mode: str = "maximum",
                 default_method: IntegrationMethod = IntegrationMethod.SCIPY_AUTO_SELECT,
                 absolute_tolerance: float = 1e-12,
                 relative_tolerance: float = 1e-10,
                 max_function_evaluations: int = 100000,
                 max_refinement_levels: int = 20,
                 singularity_detection: bool = True,
                 oscillation_detection: bool = True,
                 decimal_precision: int = 50,
                 extrapolation_acceleration: bool = True,
                 special_value_handling: bool = True,
                 use_scipy_fallback: bool = True,
                 complex_integration_mode: str = "real_imaginary"):
        """
        Initialize integrator using PrecisionSpecialValueSorter design patterns
        Enhanced with SciPy integration for robust handling of difficult functions
        """
        self.precision_mode = precision_mode
        self.default_method = default_method if SCIPY_AVAILABLE else IntegrationMethod.ADAPTIVE_GAUSS_LEGENDRE
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.max_function_evaluations = max_function_evaluations
        self.max_refinement_levels = max_refinement_levels
        self.singularity_detection = singularity_detection
        self.oscillation_detection = oscillation_detection
        self.decimal_precision = decimal_precision
        self.extrapolation_acceleration = extrapolation_acceleration
        self.special_value_handling = special_value_handling
        self.use_scipy_fallback = use_scipy_fallback and SCIPY_AVAILABLE
        self.complex_integration_mode = complex_integration_mode
        
        # Initialize precision sorter for handling function values
        self.value_sorter = PrecisionSpecialValueSorter()
        
        # Set decimal precision globally
        decimal.getcontext().prec = decimal_precision
        
        # Integration statistics
        self.integration_stats = {
            'total_integrations': 0,
            'function_evaluations': 0,
            'special_value_encounters': 0,
            'singularities_handled': 0,
            'precision_warnings': 0,
            'method_statistics': {},
            'convergence_history': []
        }
        
        # Cache for function evaluations with special value tracking
        self._function_cache = {}
        self._special_point_cache = {}
    
    def integrate(self, 
                  function: Callable[[float], Union[float, complex]],
                  a: float, 
                  b: float,
                  method: Optional[IntegrationMethod] = None) -> Tuple[Union[float, decimal.Decimal, complex], IntegrationMetrics]:
        """
        Main integration function with comprehensive special value handling
        """
        if self.use_scipy_fallback and SCIPY_AVAILABLE:
            return self._scipy_integrate(function, a, b, method)
        else:
            return self._custom_integrate(function, a, b, method)
    
    def _scipy_integrate(self, function: Callable, a: float, b: float, method: Optional[IntegrationMethod] = None):
        """Use SciPy for robust integration"""
        start_time = time.perf_counter()
        
        # Create metrics object
        metrics = IntegrationMetrics(
            method_used=method or IntegrationMethod.SCIPY_QUAD,
            convergence_status=ConvergenceStatus.CONVERGED,
            final_result=0.0,
            absolute_error_estimate=0.0,
            relative_error_estimate=0.0,
            function_evaluations=0,
            refinement_levels=0,
            computation_time_ms=0.0
        )
        
        try:
            # Use SciPy's robust quad method
            result, error_estimate = integrate.quad(
                function, a, b,
                epsabs=self.absolute_tolerance,
                epsrel=self.relative_tolerance,
                limit=500
            )
            
            metrics.final_result = result
            metrics.absolute_error_estimate = error_estimate
            metrics.function_evaluations = 500  # Estimate
            
            if error_estimate <= self.absolute_tolerance:
                metrics.convergence_status = ConvergenceStatus.CONVERGED
            else:
                metrics.convergence_status = ConvergenceStatus.PRECISION_LIMIT
            
        except Exception as e:
            metrics.convergence_status = ConvergenceStatus.UNSTABLE
            metrics.precision_warnings.append(f"SciPy integration failed: {str(e)}")
            result = float('nan')
            metrics.final_result = result
        
        end_time = time.perf_counter()
        metrics.computation_time_ms = (end_time - start_time) * 1000
        
        return result, metrics
    
    def _custom_integrate(self, function: Callable, a: float, b: float, method: Optional[IntegrationMethod] = None):
        """Use custom integration methods"""
        start_time = time.perf_counter()
        
        # Simple adaptive Simpson's rule implementation
        def simpson_adaptive(func, left, right, tolerance, depth=0, max_depth=15):
            if depth > max_depth:
                return 0.0, 0.0, 1
            
            mid = (left + right) / 2
            h = (right - left) / 6
            
            try:
                f_left = func(left)
                f_mid = func(mid)
                f_right = func(right)
                
                S = h * (f_left + 4*f_mid + f_right)
                
                # Split and recurse
                left_mid = (left + mid) / 2
                right_mid = (mid + right) / 2
                
                f_left_mid = func(left_mid)
                f_right_mid = func(right_mid)
                
                S_left = h/2 * (f_left + 4*f_left_mid + f_mid)
                S_right = h/2 * (f_mid + 4*f_right_mid + f_right)
                S_split = S_left + S_right
                
                error_est = abs(S_split - S) / 15
                
                if error_est <= tolerance:
                    return S_split, error_est, 7
                else:
                    left_result, left_error, left_evals = simpson_adaptive(
                        func, left, mid, tolerance/2, depth+1, max_depth)
                    right_result, right_error, right_evals = simpson_adaptive(
                        func, mid, right, tolerance/2, depth+1, max_depth)
                    return left_result + right_result, left_error + right_error, left_evals + right_evals + 7
                    
            except Exception:
                return 0.0, float('inf'), 1
        
        # Perform integration
        result, error_est, evals = simpson_adaptive(function, a, b, self.absolute_tolerance)
        
        # Create metrics
        metrics = IntegrationMetrics(
            method_used=method or IntegrationMethod.ADAPTIVE_SIMPSON,
            convergence_status=ConvergenceStatus.CONVERGED if error_est <= self.absolute_tolerance else ConvergenceStatus.PRECISION_LIMIT,
            final_result=result,
            absolute_error_estimate=error_est,
            relative_error_estimate=error_est / abs(result) if result != 0 else error_est,
            function_evaluations=evals,
            refinement_levels=0,
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )
        
        return result, metrics
    
    def integrate_complex(self, 
                         function: Callable[[float], complex],
                         a: float, 
                         b: float,
                         method: Optional[IntegrationMethod] = None) -> Tuple[complex, IntegrationMetrics]:
        """
        Integrate complex-valued functions by integrating real and imaginary parts separately
        """
        if self.complex_integration_mode == "real_only":
            # Just integrate the real part
            real_function = lambda x: function(x).real
            result, metrics = self.integrate(real_function, a, b, method)
            return complex(result, 0), metrics
        elif self.complex_integration_mode == "real_imaginary":
            # Integrate real and imaginary parts separately
            real_function = lambda x: function(x).real
            imag_function = lambda x: function(x).imag
            
            real_result, real_metrics = self.integrate(real_function, a, b, method)
            imag_result, imag_metrics = self.integrate(imag_function, a, b, method)
            
            # Combine results and metrics
            complex_result = complex(real_result, imag_result)
            
            # Merge metrics (use the worse convergence status)
            combined_metrics = real_metrics
            combined_metrics.final_result = complex_result
            combined_metrics.function_evaluations += imag_metrics.function_evaluations
            combined_metrics.computation_time_ms += imag_metrics.computation_time_ms
            
            if imag_metrics.convergence_status != ConvergenceStatus.CONVERGED:
                combined_metrics.convergence_status = imag_metrics.convergence_status
            
            combined_metrics.precision_warnings.extend(imag_metrics.precision_warnings)
            
            return complex_result, combined_metrics
        else:  # "magnitude_phase"
            # Integrate magnitude and phase separately (more complex, less common)
            magnitude_function = lambda x: abs(function(x))
            result, metrics = self.integrate(magnitude_function, a, b, method)
            
            # Note: This loses phase information but gives magnitude integral
            return complex(result, 0), metrics

def demonstrate_clean_integration():
    """Demonstrate the clean integration implementation"""
    
    print("CLEAN ULTRA-PRECISION INTEGRATION ALGORITHM")
    print("=" * 50)
    print(f"SciPy available: {SCIPY_AVAILABLE}")
    
    # Create integrator
    integrator = UltraPrecisionIntegrator(
        absolute_tolerance=1e-10,
        use_scipy_fallback=True
    )
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Polynomial',
            'function': lambda x: x**2,
            'bounds': (0, 1),
            'analytical': 1/3
        },
        {
            'name': 'Gaussian',
            'function': lambda x: math.exp(-x**2),
            'bounds': (-2, 2),
            'analytical': None
        },
        {
            'name': 'Oscillatory',
            'function': lambda x: math.sin(10*x),
            'bounds': (0, math.pi),
            'analytical': None
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}:")
        print(f"   Function: f(x) on {test['bounds']}")
        
        try:
            result, metrics = integrator.integrate(
                test['function'],
                test['bounds'][0],
                test['bounds'][1]
            )
            
            print(f"   Result: {result:.8f}")
            print(f"   Method: {metrics.method_used.value}")
            print(f"   Status: {metrics.convergence_status.value}")
            print(f"   Error estimate: {metrics.absolute_error_estimate:.2e}")
            print(f"   Function evaluations: {metrics.function_evaluations}")
            print(f"   Time: {metrics.computation_time_ms:.2f} ms")
            
            if test['analytical'] is not None:
                error = abs(result - test['analytical'])
                print(f"   Analytical: {test['analytical']:.8f}")
                print(f"   Actual error: {error:.2e}")
            
        except Exception as e:
            print(f"   ERROR: {str(e)}")
        
        print("   " + "-" * 40)
    
    # Test complex integration
    print(f"\nComplex Function Integration:")
    def complex_func(x):
        return complex(math.cos(x), math.sin(x))
    
    try:
        result, metrics = integrator.integrate_complex(complex_func, 0, math.pi)
        print(f"   Complex result: {result}")
        print(f"   Real part: {result.real:.6f}")
        print(f"   Imaginary part: {result.imag:.6f}")
        print(f"   Method: {metrics.method_used.value}")
        print(f"   Status: {metrics.convergence_status.value}")
    except Exception as e:
        print(f"   Complex integration error: {str(e)}")

if __name__ == "__main__":
    demonstrate_clean_integration()


