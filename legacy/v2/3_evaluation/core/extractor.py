"""
Expression extraction utilities.

Extracts mathematical expressions from model outputs in different formats:
- JSON format: {"vars": [...], "ops": [...], "cons": "C", "expr": "..."}
- Marker format: <|startofex|>...<|endofex|>
- Raw format: Direct expression string
"""

import re
import json
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of expression extraction."""

    expression: Optional[str]
    format_detected: str  # 'json', 'markers', 'raw', 'none'
    raw_output: str
    metadata: Optional[dict] = None  # vars, ops, cons from JSON format
    error: Optional[str] = None


class ExpressionExtractor:
    """Extracts expressions from model outputs."""

    # Regex patterns for different formats
    JSON_PATTERN = re.compile(
        r'\{\s*"vars"\s*:\s*\[.*?\]\s*,\s*"ops"\s*:\s*\[.*?\]\s*,\s*"cons"\s*:\s*"[^"]*"\s*,\s*"expr"\s*:\s*"([^"]*)"\s*\}',
        re.DOTALL,
    )

    # Simpler JSON pattern for partial matches
    EXPR_FIELD_PATTERN = re.compile(r'"expr"\s*:\s*"([^"]*)"')

    # Marker patterns
    START_MARKER = "<|startofex|>"
    END_MARKER = "<|endofex|>"

    def __init__(self, default_format: str = "json"):
        """
        Initialize the extractor.

        Args:
            default_format: Default format to try first ('json', 'markers', 'raw').
        """
        self.default_format = default_format

    def extract(self, output: str, format_hint: Optional[str] = None) -> ExtractionResult:
        """
        Extract expression from model output.

        Args:
            output: Raw model output string.
            format_hint: Optional hint about the expected format.

        Returns:
            ExtractionResult with extracted expression and metadata.
        """
        format_to_try = format_hint or self.default_format

        # Try formats in order
        if format_to_try == "json":
            result = self._try_json(output)
            if result.expression:
                return result
            result = self._try_markers(output)
            if result.expression:
                return result
        elif format_to_try == "markers":
            result = self._try_markers(output)
            if result.expression:
                return result
            result = self._try_json(output)
            if result.expression:
                return result
        else:
            # Try all formats
            result = self._try_json(output)
            if result.expression:
                return result
            result = self._try_markers(output)
            if result.expression:
                return result

        # No expression found
        return ExtractionResult(
            expression=None,
            format_detected="none",
            raw_output=output,
            error="Could not extract expression from output",
        )

    def _try_json(self, output: str) -> ExtractionResult:
        """Try to extract expression from JSON format."""
        # Try full JSON pattern first
        match = self.JSON_PATTERN.search(output)
        if match:
            expr = match.group(1)
            # Try to parse the full JSON for metadata
            metadata = self._parse_json_metadata(output)
            return ExtractionResult(
                expression=expr.strip() if expr else None,
                format_detected="json",
                raw_output=output,
                metadata=metadata,
            )

        # Try simpler expr field pattern
        match = self.EXPR_FIELD_PATTERN.search(output)
        if match:
            expr = match.group(1)
            metadata = self._parse_json_metadata(output)
            return ExtractionResult(
                expression=expr.strip() if expr else None,
                format_detected="json",
                raw_output=output,
                metadata=metadata,
            )

        return ExtractionResult(
            expression=None,
            format_detected="none",
            raw_output=output,
        )

    def _try_markers(self, output: str) -> ExtractionResult:
        """Try to extract expression from marker format."""
        start_idx = output.find(self.START_MARKER)
        end_idx = output.find(self.END_MARKER)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            expr = output[start_idx + len(self.START_MARKER) : end_idx]
            return ExtractionResult(
                expression=expr.strip() if expr else None,
                format_detected="markers",
                raw_output=output,
            )

        return ExtractionResult(
            expression=None,
            format_detected="none",
            raw_output=output,
        )

    def _parse_json_metadata(self, output: str) -> Optional[dict]:
        """Parse metadata (vars, ops, cons) from JSON output."""
        try:
            # Find the JSON object
            start = output.find("{")
            end = output.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            json_str = output[start:end]
            data = json.loads(json_str)

            return {
                "vars": data.get("vars", []),
                "ops": data.get("ops", []),
                "cons": data.get("cons", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return None

    def extract_batch(self, outputs: list[str], format_hint: Optional[str] = None) -> list[ExtractionResult]:
        """
        Extract expressions from multiple outputs.

        Args:
            outputs: List of raw model outputs.
            format_hint: Optional format hint.

        Returns:
            List of ExtractionResults.
        """
        return [self.extract(output, format_hint) for output in outputs]


def extract_expression(output: str, format_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[dict]]:
    """
    Convenience function to extract expression.

    Args:
        output: Raw model output.
        format_hint: Optional format hint.

    Returns:
        Tuple of (expression_string, metadata_dict).
    """
    extractor = ExpressionExtractor()
    result = extractor.extract(output, format_hint)
    return result.expression, result.metadata
