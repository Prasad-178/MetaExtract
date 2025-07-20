"""
Validation & Confidence Engine

Validates extracted data against JSON schemas and provides confidence scoring.
Identifies low-confidence fields that require human review.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"        # All schema rules must be satisfied
    LENIENT = "lenient"      # Minor violations allowed
    PERMISSIVE = "permissive"  # Major violations allowed, focus on structure


class ConfidenceLevel(Enum):
    """Confidence levels for extracted fields"""
    VERY_HIGH = 0.9  # 90%+ confidence
    HIGH = 0.75      # 75-90% confidence
    MEDIUM = 0.5     # 50-75% confidence
    LOW = 0.25       # 25-50% confidence
    VERY_LOW = 0.1   # <25% confidence


class ViolationType(Enum):
    """Types of schema violations"""
    MISSING_REQUIRED = "missing_required"
    TYPE_MISMATCH = "type_mismatch"
    FORMAT_VIOLATION = "format_violation"
    PATTERN_MISMATCH = "pattern_mismatch"
    ENUM_VIOLATION = "enum_violation"
    RANGE_VIOLATION = "range_violation"
    ADDITIONAL_PROPERTY = "additional_property"
    ARRAY_VIOLATION = "array_violation"
    CONDITIONAL_VIOLATION = "conditional_violation"


@dataclass
class ValidationViolation:
    """A schema validation violation"""
    violation_type: ViolationType
    field_path: str
    expected: Any
    actual: Any
    message: str
    severity: str  # "error", "warning", "info"


@dataclass
class FieldConfidence:
    """Confidence assessment for a field"""
    field_path: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    reasons: List[str]
    source_text: str = ""
    extraction_method: str = ""


@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    overall_confidence: float
    violations: List[ValidationViolation]
    field_confidences: Dict[str, FieldConfidence]
    low_confidence_fields: List[str]
    human_review_required: bool
    validation_summary: str


class ValidationEngine:
    """Validates extracted data and provides confidence scoring"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.confidence_threshold = 0.6  # Below this threshold = low confidence
        
        # Patterns for common formats
        self.format_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'phone': re.compile(r'^\+?[\d\s\-\(\)]{10,}$'),
            'date': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            'date-time': re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
            'ipv4': re.compile(r'^(\d{1,3}\.){3}\d{1,3}$'),
            'semantic-version': re.compile(r'^v?\d+\.\d+\.\d+(-[\w\.\-]+)?(\+[\w\.\-]+)?$')
        }
    
    def validate_extraction(self, 
                          extracted_data: Dict[str, Any],
                          schema: Dict[str, Any],
                          source_text: str = "",
                          extraction_metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate extracted data against schema and assess confidence
        
        Args:
            extracted_data: The data extracted by the system
            schema: JSON schema to validate against
            source_text: Original text the data was extracted from
            extraction_metadata: Additional metadata about extraction process
            
        Returns:
            ValidationResult with validation status and confidence scores
        """
        logger.info("Starting validation and confidence assessment")
        
        # Step 1: Schema validation
        violations = self._validate_against_schema(extracted_data, schema)
        
        # Step 2: Confidence assessment
        field_confidences = self._assess_field_confidence(
            extracted_data, schema, source_text, extraction_metadata or {}
        )
        
        # Step 3: Identify low confidence fields
        low_confidence_fields = self._identify_low_confidence_fields(field_confidences)
        
        # Step 4: Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(field_confidences)
        is_valid = self._determine_validity(violations)
        human_review_required = self._requires_human_review(violations, low_confidence_fields)
        
        # Step 5: Generate summary
        validation_summary = self._generate_validation_summary(
            violations, field_confidences, overall_confidence
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            overall_confidence=overall_confidence,
            violations=violations,
            field_confidences=field_confidences,
            low_confidence_fields=low_confidence_fields,
            human_review_required=human_review_required,
            validation_summary=validation_summary
        )
        
        logger.info(f"Validation completed: valid={is_valid}, confidence={overall_confidence:.2f}, "
                   f"violations={len(violations)}, low_confidence={len(low_confidence_fields)}")
        
        return result
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[ValidationViolation]:
        """Validate data against JSON schema"""
        violations = []
        
        # Get schema properties
        schema_type = schema.get('type', 'object')
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        additional_properties = schema.get('additionalProperties', True)
        
        if schema_type == 'object':
            # Check required fields
            for required_field in required_fields:
                field_path = f"{path}.{required_field}" if path else required_field
                if required_field not in data:
                    violations.append(ValidationViolation(
                        violation_type=ViolationType.MISSING_REQUIRED,
                        field_path=field_path,
                        expected=f"Required field '{required_field}'",
                        actual="Missing",
                        message=f"Required field '{required_field}' is missing",
                        severity="error"
                    ))
            
            # Check each field in data
            for field_name, field_value in data.items():
                field_path = f"{path}.{field_name}" if path else field_name
                
                if field_name in properties:
                    # Validate against property schema
                    field_schema = properties[field_name]
                    field_violations = self._validate_field(field_value, field_schema, field_path)
                    violations.extend(field_violations)
                elif not additional_properties:
                    # Additional properties not allowed
                    violations.append(ValidationViolation(
                        violation_type=ViolationType.ADDITIONAL_PROPERTY,
                        field_path=field_path,
                        expected="No additional properties",
                        actual=f"Additional property '{field_name}'",
                        message=f"Additional property '{field_name}' not allowed",
                        severity="warning" if self.validation_level != ValidationLevel.STRICT else "error"
                    ))
        
        return violations
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], field_path: str) -> List[ValidationViolation]:
        """Validate a single field against its schema"""
        violations = []
        
        field_type = field_schema.get('type')
        field_format = field_schema.get('format')
        field_pattern = field_schema.get('pattern')
        field_enum = field_schema.get('enum')
        field_minimum = field_schema.get('minimum')
        field_maximum = field_schema.get('maximum')
        
        # Type validation
        if field_type and not self._check_type(value, field_type):
            violations.append(ValidationViolation(
                violation_type=ViolationType.TYPE_MISMATCH,
                field_path=field_path,
                expected=field_type,
                actual=type(value).__name__,
                message=f"Expected type '{field_type}', got '{type(value).__name__}'",
                severity="error"
            ))
        
        # Format validation
        if field_format and isinstance(value, str):
            if not self._check_format(value, field_format):
                violations.append(ValidationViolation(
                    violation_type=ViolationType.FORMAT_VIOLATION,
                    field_path=field_path,
                    expected=field_format,
                    actual=value,
                    message=f"Value '{value}' does not match format '{field_format}'",
                    severity="warning"
                ))
        
        # Pattern validation
        if field_pattern and isinstance(value, str):
            if not re.match(field_pattern, value):
                violations.append(ValidationViolation(
                    violation_type=ViolationType.PATTERN_MISMATCH,
                    field_path=field_path,
                    expected=field_pattern,
                    actual=value,
                    message=f"Value '{value}' does not match pattern '{field_pattern}'",
                    severity="warning"
                ))
        
        # Enum validation
        if field_enum and value not in field_enum:
            violations.append(ValidationViolation(
                violation_type=ViolationType.ENUM_VIOLATION,
                field_path=field_path,
                expected=field_enum,
                actual=value,
                message=f"Value '{value}' is not in allowed enum values",
                severity="error"
            ))
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            if field_minimum is not None and value < field_minimum:
                violations.append(ValidationViolation(
                    violation_type=ViolationType.RANGE_VIOLATION,
                    field_path=field_path,
                    expected=f">= {field_minimum}",
                    actual=value,
                    message=f"Value {value} is below minimum {field_minimum}",
                    severity="warning"
                ))
            
            if field_maximum is not None and value > field_maximum:
                violations.append(ValidationViolation(
                    violation_type=ViolationType.RANGE_VIOLATION,
                    field_path=field_path,
                    expected=f"<= {field_maximum}",
                    actual=value,
                    message=f"Value {value} is above maximum {field_maximum}",
                    severity="warning"
                ))
        
        # Recursive validation for objects and arrays
        if field_type == 'object' and isinstance(value, dict):
            nested_violations = self._validate_against_schema(value, field_schema, field_path)
            violations.extend(nested_violations)
        
        elif field_type == 'array' and isinstance(value, list):
            items_schema = field_schema.get('items', {})
            for i, item in enumerate(value):
                item_path = f"{field_path}[{i}]"
                item_violations = self._validate_field(item, items_schema, item_path)
                violations.extend(item_violations)
        
        return violations
    
    def _assess_field_confidence(self, 
                                data: Dict[str, Any], 
                                schema: Dict[str, Any],
                                source_text: str,
                                extraction_metadata: Dict[str, Any]) -> Dict[str, FieldConfidence]:
        """Assess confidence for each extracted field"""
        field_confidences = {}
        
        def assess_recursive(obj: Any, schema_part: Dict[str, Any], path: str = ""):
            if not isinstance(obj, dict):
                return
            
            properties = schema_part.get('properties', {})
            
            for field_name, field_value in obj.items():
                field_path = f"{path}.{field_name}" if path else field_name
                field_schema = properties.get(field_name, {})
                
                confidence = self._calculate_field_confidence(
                    field_value, field_schema, field_path, source_text, extraction_metadata
                )
                
                field_confidences[field_path] = confidence
                
                # Recursive assessment for nested objects
                if isinstance(field_value, dict) and field_schema.get('type') == 'object':
                    assess_recursive(field_value, field_schema, field_path)
                
                elif isinstance(field_value, list) and field_schema.get('type') == 'array':
                    items_schema = field_schema.get('items', {})
                    for i, item in enumerate(field_value):
                        if isinstance(item, dict):
                            assess_recursive(item, items_schema, f"{field_path}[{i}]")
        
        assess_recursive(data, schema)
        return field_confidences
    
    def _calculate_field_confidence(self, 
                                   value: Any, 
                                   field_schema: Dict[str, Any],
                                   field_path: str,
                                   source_text: str,
                                   extraction_metadata: Dict[str, Any]) -> FieldConfidence:
        """Calculate confidence for a specific field"""
        confidence_score = 0.5  # Base confidence
        reasons = []
        
        # Factor 1: Value presence and completeness
        if value is None or value == "":
            confidence_score = 0.1
            reasons.append("Empty or null value")
        else:
            confidence_score += 0.2
            reasons.append("Value is present")
        
        # Factor 2: Type correctness
        expected_type = field_schema.get('type')
        if expected_type and self._check_type(value, expected_type):
            confidence_score += 0.2
            reasons.append(f"Correct type ({expected_type})")
        elif expected_type:
            confidence_score -= 0.1
            reasons.append(f"Type mismatch (expected {expected_type})")
        
        # Factor 3: Format compliance
        field_format = field_schema.get('format')
        if field_format and isinstance(value, str):
            if self._check_format(value, field_format):
                confidence_score += 0.1
                reasons.append(f"Format compliant ({field_format})")
            else:
                confidence_score -= 0.1
                reasons.append(f"Format non-compliant ({field_format})")
        
        # Factor 4: Source text evidence
        if source_text and isinstance(value, str) and value.strip():
            # Check if the value appears in source text
            if self._find_value_in_source(str(value), source_text):
                confidence_score += 0.15
                reasons.append("Value found in source text")
            else:
                confidence_score -= 0.1
                reasons.append("Value not clearly found in source")
        
        # Factor 5: Pattern matching
        field_pattern = field_schema.get('pattern')
        if field_pattern and isinstance(value, str):
            if re.match(field_pattern, value):
                confidence_score += 0.1
                reasons.append("Pattern matched")
            else:
                confidence_score -= 0.1
                reasons.append("Pattern not matched")
        
        # Factor 6: Enum membership
        field_enum = field_schema.get('enum')
        if field_enum:
            if value in field_enum:
                confidence_score += 0.15
                reasons.append("Valid enum value")
            else:
                confidence_score -= 0.2
                reasons.append("Invalid enum value")
        
        # Factor 7: Required field bonus
        if field_schema.get('required', False):
            confidence_score += 0.05
            reasons.append("Required field")
        
        # Factor 8: Extraction method quality
        extraction_method = extraction_metadata.get('method', 'unknown')
        if extraction_method in ['direct_match', 'pattern_extraction']:
            confidence_score += 0.1
            reasons.append(f"High-quality extraction method ({extraction_method})")
        elif extraction_method in ['llm_inference', 'fuzzy_match']:
            confidence_score -= 0.05
            reasons.append(f"Lower-confidence extraction method ({extraction_method})")
        
        # Normalize confidence score
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine confidence level
        if confidence_score >= 0.9:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.25:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        return FieldConfidence(
            field_path=field_path,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            reasons=reasons,
            source_text=self._extract_relevant_source_text(str(value), source_text) if value else "",
            extraction_method=extraction_method
        )
    
    def _identify_low_confidence_fields(self, field_confidences: Dict[str, FieldConfidence]) -> List[str]:
        """Identify fields with low confidence that need human review"""
        low_confidence_fields = []
        
        for field_path, confidence in field_confidences.items():
            if confidence.confidence_score < self.confidence_threshold:
                low_confidence_fields.append(field_path)
        
        return low_confidence_fields
    
    def _calculate_overall_confidence(self, field_confidences: Dict[str, FieldConfidence]) -> float:
        """Calculate overall confidence score"""
        if not field_confidences:
            return 0.0
        
        # Weighted average based on field importance
        total_score = 0.0
        total_weight = 0.0
        
        for field_path, confidence in field_confidences.items():
            # Higher weight for root-level fields
            weight = 1.0 if '.' not in field_path else 0.7
            
            total_score += confidence.confidence_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_validity(self, violations: List[ValidationViolation]) -> bool:
        """Determine if data is valid based on violations and validation level"""
        if self.validation_level == ValidationLevel.STRICT:
            return not any(v.severity == "error" for v in violations)
        elif self.validation_level == ValidationLevel.LENIENT:
            critical_errors = [v for v in violations if v.severity == "error" and v.violation_type in [
                ViolationType.MISSING_REQUIRED, ViolationType.TYPE_MISMATCH
            ]]
            return len(critical_errors) == 0
        else:  # PERMISSIVE
            return True
    
    def _requires_human_review(self, violations: List[ValidationViolation], low_confidence_fields: List[str]) -> bool:
        """Determine if human review is required"""
        # Human review required if:
        # 1. Any critical violations
        critical_violations = [v for v in violations if v.severity == "error"]
        
        # 2. Many low confidence fields
        high_low_confidence_ratio = len(low_confidence_fields) > 3
        
        # 3. Missing required fields
        missing_required = any(v.violation_type == ViolationType.MISSING_REQUIRED for v in violations)
        
        return len(critical_violations) > 0 or high_low_confidence_ratio or missing_required
    
    def _generate_validation_summary(self, 
                                   violations: List[ValidationViolation],
                                   field_confidences: Dict[str, FieldConfidence],
                                   overall_confidence: float) -> str:
        """Generate human-readable validation summary"""
        summary = f"Validation Summary:\n"
        summary += f"==================\n"
        summary += f"Overall Confidence: {overall_confidence:.2f}\n"
        summary += f"Total Fields: {len(field_confidences)}\n"
        summary += f"Violations: {len(violations)}\n\n"
        
        if violations:
            summary += f"Schema Violations:\n"
            for violation in violations[:5]:  # Show first 5
                summary += f"  ❌ {violation.field_path}: {violation.message}\n"
            if len(violations) > 5:
                summary += f"  ... and {len(violations) - 5} more violations\n"
            summary += "\n"
        
        # Confidence breakdown
        confidence_breakdown = {level: 0 for level in ConfidenceLevel}
        for confidence in field_confidences.values():
            confidence_breakdown[confidence.confidence_level] += 1
        
        summary += f"Confidence Breakdown:\n"
        for level, count in confidence_breakdown.items():
            if count > 0:
                summary += f"  {level.name}: {count} fields\n"
        
        # Low confidence fields
        low_confidence = [f for f, c in field_confidences.items() if c.confidence_score < self.confidence_threshold]
        if low_confidence:
            summary += f"\nLow Confidence Fields ({len(low_confidence)}):\n"
            for field in low_confidence[:5]:
                confidence = field_confidences[field]
                summary += f"  🚨 {field}: {confidence.confidence_score:.2f} ({', '.join(confidence.reasons[:2])})\n"
            if len(low_confidence) > 5:
                summary += f"  ... and {len(low_confidence) - 5} more fields\n"
        
        return summary
    
    # Helper methods
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        elif expected_type == 'null':
            return value is None
        else:
            return True  # Unknown type, assume valid
    
    def _check_format(self, value: str, format_name: str) -> bool:
        """Check if string value matches format"""
        pattern = self.format_patterns.get(format_name)
        if pattern:
            return bool(pattern.match(value))
        return True  # Unknown format, assume valid
    
    def _find_value_in_source(self, value: str, source_text: str) -> bool:
        """Check if value can be found in source text"""
        if not value or not source_text:
            return False
        
        # Direct match
        if value.lower() in source_text.lower():
            return True
        
        # Fuzzy match for longer values
        if len(value) > 10:
            words = value.split()
            if len(words) > 1:
                # Check if most words are present
                found_words = sum(1 for word in words if word.lower() in source_text.lower())
                return found_words / len(words) > 0.7
        
        return False
    
    def _extract_relevant_source_text(self, value: str, source_text: str, context_size: int = 50) -> str:
        """Extract relevant portion of source text around the value"""
        if not value or not source_text:
            return ""
        
        # Find value in source text
        lower_source = source_text.lower()
        lower_value = value.lower()
        
        index = lower_source.find(lower_value)
        if index == -1:
            return ""
        
        # Extract context around the value
        start = max(0, index - context_size)
        end = min(len(source_text), index + len(value) + context_size)
        
        context = source_text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(source_text):
            context = context + "..."
        
        return context
    
    def get_human_review_report(self, result: ValidationResult) -> str:
        """Generate a focused report for human review"""
        if not result.human_review_required:
            return "✅ No human review required. All validations passed with sufficient confidence."
        
        report = "🔍 Human Review Required\n"
        report += "========================\n\n"
        
        # Critical issues
        critical_violations = [v for v in result.violations if v.severity == "error"]
        if critical_violations:
            report += "❌ Critical Issues:\n"
            for violation in critical_violations:
                report += f"  • {violation.field_path}: {violation.message}\n"
            report += "\n"
        
        # Low confidence fields
        if result.low_confidence_fields:
            report += "🚨 Low Confidence Fields:\n"
            for field_path in result.low_confidence_fields:
                confidence = result.field_confidences[field_path]
                report += f"  • {field_path}: {confidence.confidence_score:.2f}\n"
                report += f"    Reasons: {', '.join(confidence.reasons[:3])}\n"
                if confidence.source_text:
                    report += f"    Source: {confidence.source_text[:100]}...\n"
                report += "\n"
        
        # Recommendations
        report += "💡 Recommendations:\n"
        if critical_violations:
            report += "  • Fix critical schema violations before proceeding\n"
        if result.low_confidence_fields:
            report += "  • Manually verify low-confidence field values\n"
            report += "  • Consider improving extraction prompts for these fields\n"
        
        report += f"\nOverall Confidence: {result.overall_confidence:.2f}\n"
        
        return report 