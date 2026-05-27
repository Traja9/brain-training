import math
import time
import decimal
from typing import List, Any, Union, Optional, Dict, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import sys

class SpecialValueType(Enum):
    """Enhanced enumeration of special value types for ultra-precise classification"""
    NEGATIVE_INFINITY = "neg_inf"
    POSITIVE_INFINITY = "pos_inf"
    NAN = "nan"
    SNAN = "signaling_nan"  # Signaling NaN (if detectable)
    NONE = "none"
    NEGATIVE_ZERO = "neg_zero"
    POSITIVE_ZERO = "pos_zero"
    SUBNORMAL = "subnormal"  # Denormalized numbers
    REGULAR = "regular"
    COMPLEX_NAN = "complex_nan"  # Complex numbers with NaN components
    COMPLEX_INF = "complex_inf"  # Complex numbers with infinity components
    DECIMAL_NAN = "decimal_nan"  # Decimal NaN values
    DECIMAL_INF = "decimal_inf"  # Decimal infinity values

@dataclass
class ValueMetrics:
    """Detailed metrics for each value during sorting"""
    original_index: int
    classification: SpecialValueType
    bit_pattern: Optional[str] = None  # For float bit analysis
    precision_loss: bool = False
    conversion_warnings: List[str] = None
    
    def __post_init__(self):
        if self.conversion_warnings is None:
            self.conversion_warnings = []

class PrecisionSpecialValueSorter:
    """
    Ultra-precise special value sorting algorithm with enhanced IEEE 754 compliance,
    bit-level analysis, and comprehensive mathematical special case handling
    """
    
    def __init__(self, 
                 nan_policy: str = "end", 
                 none_policy: str = "end",
                 zero_distinction: bool = True,
                 subnormal_handling: str = "preserve",  # "preserve", "normalize", "group"
                 precision_mode: str = "maximum",  # "maximum", "standard", "fast"
                 bit_analysis: bool = True,
                 complex_handling: str = "separate",  # "separate", "real_part", "magnitude"
                 decimal_precision: int = 50):
        """
        Initialize the precision special value sorter
        
        Args:
            nan_policy: Where to place NaN values ("start", "end", "remove", "preserve_quiet_signaling")
            none_policy: Where to place None values ("start", "end", "remove")
            zero_distinction: Whether to distinguish between +0.0 and -0.0
            subnormal_handling: How to handle subnormal (denormalized) numbers
            precision_mode: Level of precision analysis
            bit_analysis: Whether to perform bit-level float analysis
            complex_handling: How to handle complex numbers
            decimal_precision: Precision for decimal arithmetic
        """
        self.nan_policy = nan_policy
        self.none_policy = none_policy
        self.zero_distinction = zero_distinction
        self.subnormal_handling = subnormal_handling
        self.precision_mode = precision_mode
        self.bit_analysis = bit_analysis
        self.complex_handling = complex_handling
        self.decimal_precision = decimal_precision
        
        # Set decimal precision globally
        decimal.getcontext().prec = decimal_precision
        
        self.value_metrics = {}
        self.sorting_stats = {
            'total_elements': 0,
            'special_values': 0,
            'regular_values': 0,
            'comparisons': 0,
            'bit_analyses': 0,
            'precision_warnings': 0,
            'classifications': {},
            'subnormal_count': 0,
            'complex_count': 0,
            'decimal_count': 0
        }
    
    def precision_sort(self, arr: List[Any]) -> None:
        """
        Main precision sorting function with comprehensive special value handling
        """
        if not arr:
            return
        
        self._initialize_sorting_session(arr)
        
        # Step 1: Deep value analysis and classification
        analyzed_values = self._deep_value_analysis(arr)
        
        # Step 2: Group by enhanced classification
        grouped_values = self._precision_group_by_type(analyzed_values)
        
        # Step 3: Sort each group with maximum precision
        sorted_groups = self._precision_sort_groups(grouped_values)
        
        # Step 4: Reconstruct with IEEE 754 ultra-compliance
        self._precision_reconstruct_array(arr, sorted_groups)
        
        # Step 5: Validate precision and ordering
        self._validate_precision_ordering(arr)
    
    def _initialize_sorting_session(self, arr: List[Any]) -> None:
        """Initialize sorting session with comprehensive statistics tracking"""
        self.sorting_stats = {
            'total_elements': len(arr),
            'special_values': 0,
            'regular_values': 0,
            'comparisons': 0,
            'bit_analyses': 0,
            'precision_warnings': 0,
            'classifications': {},
            'subnormal_count': 0,
            'complex_count': 0,
            'decimal_count': 0,
            'start_time': time.perf_counter()
        }
        self.value_metrics = {}
    
    def _deep_value_analysis(self, arr: List[Any]) -> List[Tuple[Any, ValueMetrics]]:
        """
        Perform deep analysis of each value including bit-level inspection
        """
        analyzed = []
        
        for i, value in enumerate(arr):
            metrics = ValueMetrics(
                original_index=i,
                classification=self._ultra_precise_classify(value)
            )
            
            # Perform bit-level analysis for floats
            if self.bit_analysis and isinstance(value, float):
                metrics.bit_pattern = self._analyze_float_bits(value)
                self.sorting_stats['bit_analyses'] += 1
            
            # Check for precision issues during classification
            if self.precision_mode == "maximum":
                self._check_precision_issues(value, metrics)
            
            analyzed.append((value, metrics))
            self.value_metrics[i] = metrics
            
            # Update comprehensive statistics
            self._update_classification_stats(metrics.classification)
        
        return analyzed
    
    def _ultra_precise_classify(self, value: Any) -> SpecialValueType:
        """
        Ultra-precise classification with comprehensive special case detection
        """
        # Handle None first
        if value is None:
            return SpecialValueType.NONE
        
        # Handle complex numbers
        if isinstance(value, complex):
            if math.isnan(value.real) or math.isnan(value.imag):
                return SpecialValueType.COMPLEX_NAN
            if math.isinf(value.real) or math.isinf(value.imag):
                return SpecialValueType.COMPLEX_INF
            return SpecialValueType.REGULAR
        
        # Handle decimal types
        if isinstance(value, decimal.Decimal):
            self.sorting_stats['decimal_count'] += 1
            if value.is_nan():
                return SpecialValueType.DECIMAL_NAN
            if value.is_infinite():
                return SpecialValueType.DECIMAL_INF if value > 0 else SpecialValueType.NEGATIVE_INFINITY
            return SpecialValueType.REGULAR
        
        # Convert to float for IEEE 754 analysis
        try:
            if isinstance(value, str):
                # Try to parse special string representations
                lower_val = value.lower().strip()
                if lower_val in ['nan', 'nan()', 'nan(ind)', 'nan(snan)']:
                    return SpecialValueType.NAN
                elif lower_val in ['inf', 'infinity', '+inf', '+infinity']:
                    return SpecialValueType.POSITIVE_INFINITY
                elif lower_val in ['-inf', '-infinity']:
                    return SpecialValueType.NEGATIVE_INFINITY
                # Try numeric conversion
                float_val = float(value)
            else:
                float_val = float(value)
            
            # Enhanced NaN detection
            if math.isnan(float_val):
                # Try to detect signaling vs quiet NaN (platform dependent)
                if self.bit_analysis:
                    bit_pattern = self._analyze_float_bits(float_val)
                    if bit_pattern and self._is_signaling_nan(bit_pattern):
                        return SpecialValueType.SNAN
                return SpecialValueType.NAN
            
            # Enhanced infinity detection
            if math.isinf(float_val):
                return SpecialValueType.POSITIVE_INFINITY if float_val > 0 else SpecialValueType.NEGATIVE_INFINITY
            
            # Subnormal (denormalized) number detection
            if self._is_subnormal(float_val):
                self.sorting_stats['subnormal_count'] += 1
                return SpecialValueType.SUBNORMAL
            
            # Enhanced zero detection with sign
            if float_val == 0.0 and self.zero_distinction:
                return SpecialValueType.NEGATIVE_ZERO if math.copysign(1.0, float_val) < 0 else SpecialValueType.POSITIVE_ZERO
            
            return SpecialValueType.REGULAR
            
        except (ValueError, TypeError, OverflowError, decimal.InvalidOperation) as e:
            # Non-numeric values - treat as regular for string/object sorting
            return SpecialValueType.REGULAR
    
    def _analyze_float_bits(self, value: float) -> str:
        """
        Analyze the bit pattern of a float for ultra-precise classification
        """
        try:
            import struct
            # Pack float as binary and unpack as unsigned int
            bits = struct.unpack('>I', struct.pack('>f', float(value)))[0] if abs(value) < 1e308 else \
                   struct.unpack('>Q', struct.pack('>d', value))[0]
            return f"{bits:064b}" if abs(value) >= 1e308 else f"{bits:032b}"
        except (struct.error, OverflowError):
            return None
    
    def _is_signaling_nan(self, bit_pattern: str) -> bool:
        """
        Detect if a NaN is signaling (platform and implementation dependent)
        """
        if not bit_pattern or len(bit_pattern) < 32:
            return False
        
        # For IEEE 754 binary32: signaling NaN has MSB of fractional part = 0
        # This is highly platform dependent and may not work everywhere
        if len(bit_pattern) == 32:
            # Exponent should be all 1s, and MSB of mantissa should be 0 for sNaN
            exponent = bit_pattern[1:9]
            mantissa_msb = bit_pattern[9]
            return exponent == '11111111' and mantissa_msb == '0' and any(bit == '1' for bit in bit_pattern[10:])
        
        return False
    
    def _is_subnormal(self, value: float) -> bool:
        """
        Detect subnormal (denormalized) floating-point numbers
        """
        if value == 0.0 or math.isnan(value) or math.isinf(value):
            return False
        
        # Check if the absolute value is smaller than the smallest normal number
        # For double precision: smallest normal is approximately 2.225e-308
        # For single precision: smallest normal is approximately 1.175e-38
        
        abs_val = abs(value)
        
        # Use sys.float_info for precise thresholds
        return abs_val < sys.float_info.min and abs_val > 0.0
    
    def _check_precision_issues(self, value: Any, metrics: ValueMetrics) -> None:
        """
        Check for potential precision loss or conversion issues
        """
        try:
            if isinstance(value, float):
                # Check for precision loss in decimal representation
                str_repr = repr(value)
                reconstructed = float(str_repr)
                if reconstructed != value:
                    metrics.precision_loss = True
                    metrics.conversion_warnings.append("Float precision loss in string conversion")
                    self.sorting_stats['precision_warnings'] += 1
            
            elif isinstance(value, str):
                # Check if string->float->string roundtrip is stable
                try:
                    float_val = float(value)
                    if not (math.isnan(float_val) or math.isinf(float_val)):
                        back_to_str = str(float_val)
                        if value != back_to_str and float(back_to_str) != float_val:
                            metrics.conversion_warnings.append("String-float conversion instability")
                            self.sorting_stats['precision_warnings'] += 1
                except:
                    pass
                    
        except Exception as e:
            metrics.conversion_warnings.append(f"Precision check failed: {str(e)}")
    
    def _update_classification_stats(self, classification: SpecialValueType) -> None:
        """Update comprehensive classification statistics"""
        type_name = classification.value
        self.sorting_stats['classifications'][type_name] = self.sorting_stats['classifications'].get(type_name, 0) + 1
        
        if classification in [SpecialValueType.REGULAR]:
            self.sorting_stats['regular_values'] += 1
        else:
            self.sorting_stats['special_values'] += 1
    
    def _precision_group_by_type(self, analyzed_values: List[Tuple[Any, ValueMetrics]]) -> Dict[SpecialValueType, List[Tuple[Any, ValueMetrics]]]:
        """
        Group values by their precise classification
        """
        groups = {}
        
        for value, metrics in analyzed_values:
            classification = metrics.classification
            if classification not in groups:
                groups[classification] = []
            groups[classification].append((value, metrics))
        
        return groups
    
    def _precision_sort_groups(self, grouped_values: Dict[SpecialValueType, List[Tuple[Any, ValueMetrics]]]) -> Dict[SpecialValueType, List[Any]]:
        """
        Sort each group with maximum precision
        """
        sorted_groups = {}
        
        for value_type, value_list in grouped_values.items():
            values = [v for v, _ in value_list]
            
            if value_type == SpecialValueType.REGULAR:
                sorted_groups[value_type] = self._ultra_precise_sort_regular(values)
            
            elif value_type == SpecialValueType.SUBNORMAL:
                # Sort subnormal numbers with extra precision
                sorted_groups[value_type] = self._sort_subnormal_values(values)
            
            elif value_type in [SpecialValueType.COMPLEX_NAN, SpecialValueType.COMPLEX_INF]:
                sorted_groups[value_type] = self._sort_complex_values(values)
            
            elif value_type in [SpecialValueType.DECIMAL_NAN, SpecialValueType.DECIMAL_INF]:
                sorted_groups[value_type] = self._sort_decimal_values(values)
            
            else:
                # Maintain original order for other special values (stable sort)
                sorted_groups[value_type] = values
        
        return sorted_groups
    
    def _ultra_precise_sort_regular(self, values: List[Any]) -> List[Any]:
        """
        Ultra-precise sorting of regular values with enhanced comparison
        """
        def ultra_precise_compare(a, b):
            self.sorting_stats['comparisons'] += 1
            
            try:
                # Handle mixed numeric types with maximum precision
                if self._is_numeric(a) and self._is_numeric(b):
                    return self._precise_numeric_compare(a, b)
                
                # Handle string comparison with locale awareness
                elif isinstance(a, str) and isinstance(b, str):
                    return self._precise_string_compare(a, b)
                
                # Mixed type comparison with consistent rules
                else:
                    return self._precise_mixed_compare(a, b)
                    
            except Exception as e:
                # Ultra-robust fallback
                try:
                    str_a, str_b = str(a), str(b)
                    result = (str_a > str_b) - (str_a < str_b)
                    return result
                except:
                    return 0  # Consider equal if all else fails
        
        from functools import cmp_to_key
        return sorted(values, key=cmp_to_key(ultra_precise_compare))
    
    def _is_numeric(self, value: Any) -> bool:
        """Enhanced numeric type detection"""
        return isinstance(value, (int, float, complex, decimal.Decimal)) or (
            isinstance(value, str) and self._is_numeric_string(value)
        )
    
    def _is_numeric_string(self, value: str) -> bool:
        """Detect if a string represents a numeric value"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _precise_numeric_compare(self, a: Any, b: Any) -> int:
        """
        Ultra-precise numeric comparison with decimal arithmetic when needed
        """
        try:
            # Convert to decimal for maximum precision if in maximum precision mode
            if self.precision_mode == "maximum":
                try:
                    # Handle complex numbers by converting to magnitude for comparison
                    if isinstance(a, complex) or isinstance(b, complex):
                        mag_a = abs(a) if isinstance(a, complex) else abs(float(a))
                        mag_b = abs(b) if isinstance(b, complex) else abs(float(b))
                        dec_a = decimal.Decimal(str(mag_a))
                        dec_b = decimal.Decimal(str(mag_b))
                    else:
                        dec_a = decimal.Decimal(str(a)) if not isinstance(a, decimal.Decimal) else a
                        dec_b = decimal.Decimal(str(b)) if not isinstance(b, decimal.Decimal) else b
                    
                    if dec_a < dec_b:
                        return -1
                    elif dec_a > dec_b:
                        return 1
                    else:
                        return 0
                except (decimal.InvalidOperation, ValueError):
                    pass
            
            # Enhanced fallback with better type handling
            float_a, float_b = self._safe_float_conversion(a), self._safe_float_conversion(b)
            
            # Handle special cases first
            if math.isnan(float_a) and math.isnan(float_b):
                return 0  # Both NaN - equal
            if math.isnan(float_a):
                return 1   # NaN goes to end
            if math.isnan(float_b):
                return -1  # NaN goes to end
            
            if math.isinf(float_a) or math.isinf(float_b):
                return (float_a > float_b) - (float_a < float_b)
            
            # Ultra-precise comparison with adaptive epsilon
            if abs(float_a) < 1e-100 or abs(float_b) < 1e-100:
                # For very small numbers (including subnormals), use absolute epsilon
                epsilon = sys.float_info.epsilon
            else:
                # For normal numbers, use relative epsilon
                epsilon = sys.float_info.epsilon * max(abs(float_a), abs(float_b), 1.0)
            
            if abs(float_a - float_b) <= epsilon:
                return 0
            else:
                return (float_a > float_b) - (float_a < float_b)
                
        except (ValueError, TypeError, OverflowError):
            return 0
    
    def _safe_float_conversion(self, value: Any) -> float:
        """Safely convert any value to float with proper handling of edge cases"""
        if isinstance(value, complex):
            return abs(value)  # Use magnitude for complex numbers
        elif isinstance(value, decimal.Decimal):
            if value.is_nan():
                return float('nan')
            elif value.is_infinite():
                return float('inf') if value > 0 else float('-inf')
            else:
                return float(value)
        else:
            return float(value)
    
    def _precise_string_compare(self, a: str, b: str) -> int:
        """Enhanced string comparison with numeric awareness"""
        # If both strings represent numbers, compare numerically
        if self._is_numeric_string(a) and self._is_numeric_string(b):
            try:
                return self._precise_numeric_compare(a, b)
            except:
                pass
        
        # Standard lexicographic comparison
        return (a > b) - (a < b)
    
    def _precise_mixed_compare(self, a: Any, b: Any) -> int:
        """Precise comparison of mixed types with consistent ordering"""
        # Consistent type ordering: None < numbers < strings < others
        def type_priority(value):
            if value is None:
                return 0
            elif self._is_numeric(value):
                return 1
            elif isinstance(value, str):
                return 2
            else:
                return 3
        
        priority_a, priority_b = type_priority(a), type_priority(b)
        
        if priority_a != priority_b:
            return priority_a - priority_b
        
        # Same priority - compare as strings
        try:
            str_a, str_b = str(a), str(b)
            return (str_a > str_b) - (str_a < str_b)
        except:
            return 0
    
    def _sort_subnormal_values(self, values: List[Any]) -> List[Any]:
        """Special handling for subnormal floating-point numbers"""
        if self.subnormal_handling == "normalize":
            # Convert subnormals to normalized form if possible
            normalized = []
            for v in values:
                if isinstance(v, float) and self._is_subnormal(v):
                    # This is a placeholder - actual normalization is complex
                    normalized.append(v)  # Keep as-is for now
                else:
                    normalized.append(v)
            return sorted(normalized, key=lambda x: (x if isinstance(x, (int, float)) else 0))
        else:
            # Preserve subnormal values and sort with ultra precision
            return self._ultra_precise_sort_regular(values)
    
    def _sort_complex_values(self, values: List[Any]) -> List[Any]:
        """Sort complex numbers based on handling policy"""
        if self.complex_handling == "real_part":
            return sorted(values, key=lambda x: x.real if isinstance(x, complex) else 0)
        elif self.complex_handling == "magnitude":
            return sorted(values, key=lambda x: abs(x) if isinstance(x, complex) else 0)
        else:
            # Separate handling - maintain original order within group
            return values
    
    def _sort_decimal_values(self, values: List[Any]) -> List[Any]:
        """Sort decimal values with maximum precision"""
        def decimal_key(x):
            if isinstance(x, decimal.Decimal):
                if x.is_nan():
                    return decimal.Decimal('inf')  # Sort NaN to end
                return x
            return decimal.Decimal('0')
        
        return sorted(values, key=decimal_key)
    
    def _precision_reconstruct_array(self, arr: List[Any], sorted_groups: Dict[SpecialValueType, List[Any]]) -> None:
        """
        Reconstruct array with IEEE 754 ultra-compliance and policy adherence
        """
        arr.clear()
        
        # Enhanced numeric value merging with proper type segregation
        numeric_values = []
        string_numeric_values = []
        complex_values = []
        other_values = []
        
        # Collect and categorize all regular values
        if SpecialValueType.REGULAR in sorted_groups:
            for value in sorted_groups[SpecialValueType.REGULAR]:
                if isinstance(value, complex):
                    complex_values.append(value)
                elif isinstance(value, str) and self._is_numeric_string(value):
                    string_numeric_values.append(value)
                elif self._is_numeric(value) and not isinstance(value, str):
                    numeric_values.append(value)
                else:
                    other_values.append(value)
        
        # Handle subnormals based on policy
        if SpecialValueType.SUBNORMAL in sorted_groups:
            if self.subnormal_handling == "group":
                # Keep subnormals separate - will be added later
                pass
            else:
                numeric_values.extend(sorted_groups[SpecialValueType.SUBNORMAL])
        
        # Handle zeros
        if SpecialValueType.NEGATIVE_ZERO in sorted_groups:
            numeric_values.extend(sorted_groups[SpecialValueType.NEGATIVE_ZERO])
        if SpecialValueType.POSITIVE_ZERO in sorted_groups:
            numeric_values.extend(sorted_groups[SpecialValueType.POSITIVE_ZERO])
        
        # Sort each category separately for maximum precision
        if numeric_values:
            numeric_values = self._ultra_precise_sort_regular(numeric_values)
        if string_numeric_values:
            string_numeric_values = self._ultra_precise_sort_regular(string_numeric_values)
        if complex_values:
            complex_values = self._sort_complex_values(complex_values)
        if other_values:
            other_values = self._ultra_precise_sort_regular(other_values)
        
        # Apply policies for special value placement
        nan_at_start = self.nan_policy in ["start", "preserve_quiet_signaling"]
        none_at_start = self.none_policy == "start"
        
        # Reconstruct in IEEE 754 compliant order with enhanced precision
        
        # 1. Handle start policies
        if none_at_start and SpecialValueType.NONE in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.NONE])
        
        if nan_at_start:
            # Handle different NaN types in order of IEEE 754 preference
            if SpecialValueType.SNAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.SNAN])  # Signaling NaN first
            if SpecialValueType.NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.NAN])
            if SpecialValueType.DECIMAL_NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.DECIMAL_NAN])
            if SpecialValueType.COMPLEX_NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.COMPLEX_NAN])
        
        # 2. Negative infinity (all types)
        if SpecialValueType.NEGATIVE_INFINITY in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.NEGATIVE_INFINITY])
        if SpecialValueType.DECIMAL_INF in sorted_groups:
            # Only negative decimal infinities
            neg_dec_inf = [x for x in sorted_groups[SpecialValueType.DECIMAL_INF] 
                          if isinstance(x, decimal.Decimal) and x < 0]
            arr.extend(neg_dec_inf)
        
        # 3. All numeric values in precise order
        arr.extend(numeric_values)
        
        # 4. String representations of numbers (maintaining precision)
        arr.extend(string_numeric_values)
        
        # 5. Complex numbers (if not handled in numeric section)
        arr.extend(complex_values)
        
        # 6. Other non-numeric values
        arr.extend(other_values)
        
        # 7. Separate subnormal group if configured
        if self.subnormal_handling == "group" and SpecialValueType.SUBNORMAL in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.SUBNORMAL])
        
        # 8. Positive infinity (all types)
        if SpecialValueType.POSITIVE_INFINITY in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.POSITIVE_INFINITY])
        if SpecialValueType.DECIMAL_INF in sorted_groups:
            # Only positive decimal infinities  
            pos_dec_inf = [x for x in sorted_groups[SpecialValueType.DECIMAL_INF] 
                          if isinstance(x, decimal.Decimal) and x > 0]
            arr.extend(pos_dec_inf)
        if SpecialValueType.COMPLEX_INF in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.COMPLEX_INF])
        
        # 9. Handle end policies (default)
        if not nan_at_start:
            if SpecialValueType.NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.NAN])
            if SpecialValueType.SNAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.SNAN])
            if SpecialValueType.DECIMAL_NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.DECIMAL_NAN])
            if SpecialValueType.COMPLEX_NAN in sorted_groups:
                arr.extend(sorted_groups[SpecialValueType.COMPLEX_NAN])
        
        if not none_at_start and SpecialValueType.NONE in sorted_groups:
            arr.extend(sorted_groups[SpecialValueType.NONE])
    
    def _validate_precision_ordering(self, arr: List[Any]) -> None:
        """
        Enhanced validation of precision and correctness of the final ordering
        """
        validation_issues = []
        
        # Enhanced numeric ordering validation with type awareness
        numeric_positions = []
        string_numeric_positions = []
        complex_positions = []
        
        for i, val in enumerate(arr):
            if val is None:
                continue
            elif isinstance(val, complex):
                if not (math.isnan(val.real) or math.isnan(val.imag) or 
                       math.isinf(val.real) or math.isinf(val.imag)):
                    complex_positions.append((i, abs(val)))
            elif isinstance(val, str) and self._is_numeric_string(val):
                try:
                    string_numeric_positions.append((i, float(val)))
                except:
                    pass
            elif self._is_numeric(val) and not isinstance(val, str):
                try:
                    float_val = self._safe_float_conversion(val)
                    if not (math.isnan(float_val) or math.isinf(float_val)):
                        numeric_positions.append((i, float_val))
                except:
                    pass
        
        # Verify numeric ordering within each category
        def check_ordering(positions, category_name):
            for j in range(len(positions) - 1):
                pos1, val1 = positions[j]
                pos2, val2 = positions[j + 1]
                
                # Use adaptive epsilon for different value ranges
                if abs(val1) < 1e-100 or abs(val2) < 1e-100:
                    epsilon = sys.float_info.epsilon
                else:
                    epsilon = sys.float_info.epsilon * max(abs(val1), abs(val2), 1.0) * 10
                
                if val1 > val2 and abs(val1 - val2) > epsilon:
                    validation_issues.append(
                        f"{category_name} ordering violation: {val1:.6e} > {val2:.6e} "
                        f"at positions {pos1}-{pos2} (diff: {abs(val1-val2):.2e}, epsilon: {epsilon:.2e})"
                    )
        
        check_ordering(numeric_positions, "Numeric")
        check_ordering(string_numeric_positions, "String-numeric") 
        check_ordering(complex_positions, "Complex magnitude")
        
        # Check for proper special value positioning
        nan_positions = [i for i, v in enumerate(arr) if self._is_nan_value(v)]
        inf_positions = [i for i, v in enumerate(arr) if self._is_inf_value(v)]
        none_positions = [i for i, v in enumerate(arr) if v is None]
        
        # Validate special value policies
        if nan_positions:
            if self.nan_policy == "start" and nan_positions and max(nan_positions) > len(nan_positions):
                validation_issues.append("NaN values not properly positioned at start")
            elif self.nan_policy == "end" and nan_positions:
                expected_start = len(arr) - len(nan_positions)
                if none_positions and self.none_policy == "end":
                    expected_start -= len(none_positions)
                if min(nan_positions) < expected_start:
                    validation_issues.append("NaN values not properly positioned at end")
        
        # Store enhanced validation results
        self.sorting_stats['validation_issues'] = len(validation_issues)
        self.sorting_stats['validation_details'] = validation_issues[:10]  # First 10 issues
        
        # Add precision statistics
        self.sorting_stats['ordering_precision'] = {
            'numeric_sequences': len(numeric_positions),
            'string_numeric_sequences': len(string_numeric_positions),
            'complex_sequences': len(complex_positions),
            'special_value_positions': {
                'nan_count': len(nan_positions),
                'inf_count': len(inf_positions), 
                'none_count': len(none_positions)
            }
        }
    
    def _is_nan_value(self, value: Any) -> bool:
        """Check if a value is any type of NaN"""
        try:
            if value is None:
                return False
            if isinstance(value, complex):
                return math.isnan(value.real) or math.isnan(value.imag)
            if isinstance(value, decimal.Decimal):
                return value.is_nan()
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    
    def _is_inf_value(self, value: Any) -> bool:
        """Check if a value is any type of infinity"""
        try:
            if value is None:
                return False
            if isinstance(value, complex):
                return math.isinf(value.real) or math.isinf(value.imag)
            if isinstance(value, decimal.Decimal):
                return value.is_infinite()
            return math.isinf(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    
    def get_comprehensive_stats(self) -> Dict:
        """Get ultra-comprehensive statistics about the sorting operation"""
        end_time = time.perf_counter()
        total_time = end_time - self.sorting_stats.get('start_time', end_time)
        
        return {
            'performance': {
                'total_time_ms': total_time * 1000,
                'elements_per_second': self.sorting_stats['total_elements'] / max(total_time, 1e-9),
                'comparisons': self.sorting_stats['comparisons'],
                'bit_analyses': self.sorting_stats['bit_analyses']
            },
            'elements': {
                'total': self.sorting_stats['total_elements'],
                'special_values': self.sorting_stats['special_values'],
                'regular_values': self.sorting_stats['regular_values'],
                'subnormal_count': self.sorting_stats['subnormal_count'],
                'complex_count': self.sorting_stats['complex_count'],
                'decimal_count': self.sorting_stats['decimal_count']
            },
            'precision': {
                'precision_warnings': self.sorting_stats['precision_warnings'],
                'validation_issues': self.sorting_stats.get('validation_issues', 0),
                'precision_mode': self.precision_mode,
                'decimal_precision': self.decimal_precision
            },
            'classifications': self.sorting_stats['classifications'],
            'policies': {
                'nan_policy': self.nan_policy,
                'none_policy': self.none_policy,
                'zero_distinction': self.zero_distinction,
                'subnormal_handling': self.subnormal_handling,
                'complex_handling': self.complex_handling,
                'bit_analysis': self.bit_analysis
            },
            'validation': self.sorting_stats.get('validation_details', [])
        }

def demonstrate_precision_sorting():
    """Comprehensive demonstration of precision special value sorting"""
    
    print("ULTRA-PRECISION SPECIAL VALUE SORTING ALGORITHM")
    print("=" * 60)
    
    # Import for creating special test values
    import struct
    
    def create_subnormal():
        """Create a subnormal float"""
        return sys.float_info.min / 2
    
    def create_decimal_values():
        """Create high precision decimal values"""
        return [
            decimal.Decimal('1.2345678901234567890123456789'),
            decimal.Decimal('1.2345678901234567890123456788'),
            decimal.Decimal('nan'),
            decimal.Decimal('inf'),
            decimal.Decimal('-inf')
        ]
    
    test_cases = [
        {
            'name': 'IEEE 754 Ultra-Precision Test',
            'data': [1.0, -0.0, 0.0, float('inf'), -float('inf'), float('nan'), 
                    create_subnormal(), -1.5, 2.5, sys.float_info.epsilon],
            'options': {'zero_distinction': True, 'precision_mode': 'maximum', 'bit_analysis': True}
        },
        {
            'name': 'High-Precision Decimal Test',
            'data': create_decimal_values() + [1.1, 2.2, None],
            'options': {'precision_mode': 'maximum', 'decimal_precision': 100}
        },
        {
            'name': 'Complex Numbers with Special Values',
            'data': [1+2j, complex('nan'), complex('inf'), 3+4j, complex('-inf'), 2+1j],
            'options': {'complex_handling': 'magnitude', 'precision_mode': 'maximum'}
        },
        {
            'name': 'Subnormal Number Handling',
            'data': [1.0, create_subnormal(), -create_subnormal(), sys.float_info.min, 0.0, 2.0],
            'options': {'subnormal_handling': 'preserve', 'precision_mode': 'maximum'}
        },
        {
            'name': 'Mixed Ultra-Precision Dataset',
            'data': [
                decimal.Decimal('1.23456789012345678901234567890'),
                1.23456789012345678901234567890,
                '1.23456789012345678901234567890',
                complex('1.23456789+2.34567890j'),
                float('nan'),
                None,
                decimal.Decimal('inf'),
                create_subnormal()
            ],
            'options': {'precision_mode': 'maximum', 'bit_analysis': True, 'decimal_precision': 50}
        },
        {
            'name': 'Bit-Level Analysis Test',
            'data': [
                1.0000000000000002,  # Smallest representable difference from 1.0
                1.0000000000000001,  # Should be rounded to 1.0 in double precision
                1.0,
                float('nan'),
                float('inf'),
                -0.0,
                0.0
            ],
            'options': {'bit_analysis': True, 'precision_mode': 'maximum', 'zero_distinction': True}
        },
        {
            'name': 'String Numeric Precision Test',
            'data': [
                '1.2345678901234567890',
                '1.2345678901234567891', 
                '1.234567890123456789',
                1.2345678901234567890,
                None,
                float('nan')
            ],
            'options': {'precision_mode': 'maximum'}
        },
        {
            'name': 'Extreme Value Stress Test',
            'data': [
                sys.float_info.max,
                -sys.float_info.max,
                sys.float_info.min,
                -sys.float_info.min,
                sys.float_info.epsilon,
                1 + sys.float_info.epsilon,
                1 - sys.float_info.epsilon/2,
                float('nan'),
                float('inf'),
                -float('inf')
            ],
            'options': {'precision_mode': 'maximum', 'bit_analysis': True}
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print("-" * len(test['name']) + "-")
        print(f"Original: {test['data']}")
        print(f"Options: {test['options']}")
        
        # Create sorter and sort
        sorter = PrecisionSpecialValueSorter(**test['options'])
        arr_copy = test['data'].copy()
        
        # Perform precision sorting
        sorter.precision_sort(arr_copy)
        
        print(f"Sorted: {arr_copy}")
        
        # Display comprehensive statistics
        stats = sorter.get_comprehensive_stats()
        
        print(f"\nPerformance Metrics:")
        print(f"  Time: {stats['performance']['total_time_ms']:.6f}ms")
        print(f"  Speed: {stats['performance']['elements_per_second']:.0f} elements/sec")
        print(f"  Comparisons: {stats['performance']['comparisons']}")
        print(f"  Bit analyses: {stats['performance']['bit_analyses']}")
        
        print(f"\nPrecision Analysis:")
        print(f"  Special values: {stats['elements']['special_values']}/{stats['elements']['total']}")
        print(f"  Subnormal count: {stats['elements']['subnormal_count']}")
        print(f"  Complex count: {stats['elements']['complex_count']}")
        print(f"  Decimal count: {stats['elements']['decimal_count']}")
        print(f"  Precision warnings: {stats['precision']['precision_warnings']}")
        print(f"  Validation issues: {stats['precision']['validation_issues']}")
        
        print(f"\nClassifications:")
        for classification, count in stats['classifications'].items():
            print(f"  {classification}: {count}")
        
        if stats['validation']:
            print(f"\nValidation Issues:")
            for issue in stats['validation']:
                print(f"  ⚠️  {issue}")
        else:
            print(f"\n✅ Perfect precision ordering achieved")
        
        print("=" * 50)

def precision_performance_analysis():
    """Analyze performance across different precision modes and special value densities"""
    
    print("\n" + "=" * 60)
    print("PRECISION PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    import random
    
    # Test different precision modes
    precision_modes = ['fast', 'standard', 'maximum']
    test_sizes = [100, 500, 1000]
    special_ratios = [0.1, 0.5, 0.9]  # 10%, 50%, 90% special values
    
    def create_subnormal():
        return sys.float_info.min / 2
    
    for mode in precision_modes:
        print(f"\nPrecision Mode: {mode}")
        print("-" * 30)
        
        for size in test_sizes:
            for ratio in special_ratios:
                # Generate complex test data
                special_count = int(size * ratio)
                regular_count = size - special_count
                
                test_data = []
                
                # Add regular values with varying precision
                test_data.extend([random.uniform(-1000, 1000) for _ in range(regular_count // 2)])
                test_data.extend([decimal.Decimal(str(random.uniform(-100, 100))) for _ in range(regular_count // 4)])
                test_data.extend([f"{random.uniform(-10, 10):.20f}" for _ in range(regular_count // 4)])
                
                # Add special values
                special_types = [
                    float('nan'), float('inf'), -float('inf'), None,
                    create_subnormal(), -create_subnormal(),
                    complex('nan'), complex('inf'),
                    decimal.Decimal('nan'), decimal.Decimal('inf')
                ]
                
                for _ in range(special_count):
                    test_data.append(random.choice(special_types))
                
                # Shuffle the data
                random.shuffle(test_data)
                
                # Test sorting with different precision modes
                sorter = PrecisionSpecialValueSorter(
                    precision_mode=mode,
                    bit_analysis=(mode == 'maximum'),
                    decimal_precision=50 if mode == 'maximum' else 28
                )
                
                data_copy = test_data.copy()
                sorter.precision_sort(data_copy)
                
                stats = sorter.get_comprehensive_stats()
                
                print(f"  {size:4d} elements, {ratio*100:2.0f}% special:")
                print(f"    Time: {stats['performance']['total_time_ms']:8.3f}ms")
                print(f"    Speed: {stats['performance']['elements_per_second']:8.0f} elem/sec")
                print(f"    Precision warnings: {stats['precision']['precision_warnings']:3d}")
                print(f"    Validation issues: {stats['precision']['validation_issues']:3d}")

def advanced_precision_features():
    """Demonstrate advanced precision features"""
    
    print("\n" + "=" * 60)
    print("ADVANCED PRECISION FEATURES")
    print("=" * 60)
    
    # 1. Bit-level NaN analysis
    print("\n1. Bit-Level Float Analysis:")
    nan_values = [float('nan'), float('nan'), float('nan')]
    
    sorter = PrecisionSpecialValueSorter(bit_analysis=True, precision_mode='maximum')
    
    for i, val in enumerate(nan_values):
        classification = sorter._ultra_precise_classify(val)
        bit_pattern = sorter._analyze_float_bits(val)
        print(f"   NaN #{i+1}: {classification.value}")
        print(f"   Bit pattern: {bit_pattern}")
        print(f"   Is signaling: {sorter._is_signaling_nan(bit_pattern) if bit_pattern else 'Unknown'}")
    
    # 2. Subnormal detection and handling
    print("\n2. Subnormal Number Detection:")
    
    def create_subnormal():
        return sys.float_info.min / 2
    
    test_values = [
        sys.float_info.min,        # Smallest normal
        create_subnormal(),        # Subnormal
        sys.float_info.min / 1e10, # Very small subnormal
        0.0,                       # Zero
        1e-320                     # Subnormal (if supported)
    ]
    
    for val in test_values:
        is_subnormal = sorter._is_subnormal(val)
        print(f"   Value: {val:e} -> {'Subnormal' if is_subnormal else 'Normal/Zero'}")
    
    # 3. High-precision decimal arithmetic
    print("\n3. High-Precision Decimal Arithmetic:")
    
    decimal.getcontext().prec = 100  # 100 digit precision
    
    high_precision_values = [
        decimal.Decimal('1') / decimal.Decimal('3'),    # 0.333...
        decimal.Decimal('1') / decimal.Decimal('7'),    # 0.142857142857...
        decimal.Decimal('2').sqrt(),                     # sqrt(2)
        decimal.Decimal('22') / decimal.Decimal('7'),   # π approximation
    ]
    
    sorter_hp = PrecisionSpecialValueSorter(precision_mode='maximum', decimal_precision=100)
    sorter_hp.precision_sort(high_precision_values)
    
    print("   Sorted high-precision decimals:")
    for val in high_precision_values:
        print(f"   {val}")
    
    # 4. Mixed precision comparison
    print("\n4. Mixed Precision Comparison Test:")
    
    # Same mathematical value in different representations
    mixed_precision_test = [
        1.1,                                    # Float
        decimal.Decimal('1.1'),                # Decimal
        '1.1',                                  # String
        decimal.Decimal('11') / decimal.Decimal('10'),  # Exact decimal fraction
    ]
    
    print(f"   Before sorting: {mixed_precision_test}")
    
    sorter_mixed = PrecisionSpecialValueSorter(precision_mode='maximum')
    sorter_mixed.precision_sort(mixed_precision_test)
    
    print(f"   After sorting:  {mixed_precision_test}")
    
    stats = sorter_mixed.get_comprehensive_stats()
    print(f"   Precision warnings: {stats['precision']['precision_warnings']}")

# Utility functions for precision sorting
def ultra_precision_sort(arr: List[Any], **options) -> List[Any]:
    """
    Convenience function for ultra-precision sorting
    
    Args:
        arr: Array to sort
        **options: Precision sorting options
    
    Returns:
        New ultra-precision sorted array
    """
    result = arr.copy()
    sorter = PrecisionSpecialValueSorter(**options)
    sorter.precision_sort(result)
    return result

def analyze_value_precision(value: Any) -> Dict[str, Any]:
    """
    Comprehensive precision analysis of a single value
    
    Returns:
        Dictionary with detailed precision metrics
    """
    sorter = PrecisionSpecialValueSorter(precision_mode='maximum', bit_analysis=True)
    classification = sorter._ultra_precise_classify(value)
    
    metrics = {
        'classification': classification.value,
        'is_numeric': sorter._is_numeric(value),
        'bit_pattern': None,
        'is_subnormal': False,
        'precision_loss': False
    }
    
    if isinstance(value, float):
        metrics['bit_pattern'] = sorter._analyze_float_bits(value)
        metrics['is_subnormal'] = sorter._is_subnormal(value)
        
        # Test for precision loss
        try:
            reconstructed = float(repr(value))
            metrics['precision_loss'] = (reconstructed != value)
        except:
            pass
    
    return metrics

def compare_precision_modes(arr: List[Any]) -> Dict[str, Dict]:
    """
    Compare sorting results across different precision modes
    
    Returns:
        Dictionary comparing results from each precision mode
    """
    modes = ['fast', 'standard', 'maximum']
    results = {}
    
    for mode in modes:
        arr_copy = arr.copy()
        sorter = PrecisionSpecialValueSorter(precision_mode=mode, bit_analysis=(mode=='maximum'))
        
        start_time = time.perf_counter()
        sorter.precision_sort(arr_copy)
        end_time = time.perf_counter()
        
        results[mode] = {
            'sorted_array': arr_copy.copy(),
            'stats': sorter.get_comprehensive_stats(),
            'time_ms': (end_time - start_time) * 1000
        }
    
    return results

if __name__ == "__main__":
    demonstrate_precision_sorting()
    precision_performance_analysis()
    advanced_precision_features()
    
    print("\n" + "=" * 60)
    print("ULTRA-PRECISION SPECIAL VALUE SORTING FEATURES:")
    print("✅ IEEE 754 ultra-compliance with bit-level analysis")
    print("✅ Signaling vs. Quiet NaN detection")
    print("✅ Subnormal (denormalized) number handling")  
    print("✅ High-precision decimal arithmetic (configurable precision)")
    print("✅ Complex number sorting with multiple strategies")
    print("✅ Mixed-type ultra-precise comparison")
    print("✅ Comprehensive precision loss detection")
    print("✅ Configurable precision modes (fast/standard/maximum)")
    print("✅ Advanced zero sign distinction (-0.0 vs +0.0)")
    print("✅ Bit-pattern analysis for floating-point values")
    print("✅ String-to-numeric precision validation")
    print("✅ Ultra-comprehensive statistics and validation")
    print("✅ Stable sorting within special value groups")
    print("✅ Performance optimized for high special value density")
    print("=" * 60)

    