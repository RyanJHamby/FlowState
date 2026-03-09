"""Schema enforcement and sequence gap detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa


@dataclass
class ValidationError:
    """Describes a single validation failure."""

    field_name: str
    error_type: str
    message: str
    row_index: int | None = None


@dataclass
class ValidationResult:
    """Result of validating a record batch."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    rows_validated: int = 0


class SchemaValidator:
    """Validates Arrow RecordBatches against a target schema.

    Checks field presence, types, and nullability constraints.
    """

    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def validate(self, batch: pa.RecordBatch) -> ValidationResult:
        """Validate a RecordBatch against the target schema.

        Args:
            batch: The RecordBatch to validate.

        Returns:
            ValidationResult with any errors found.
        """
        errors: list[ValidationError] = []

        # Check all required fields are present
        for i in range(len(self._schema)):
            expected_field = self._schema.field(i)
            if expected_field.name not in batch.schema.names:
                errors.append(
                    ValidationError(
                        field_name=expected_field.name,
                        error_type="missing_field",
                        message=f"Required field '{expected_field.name}' not found in batch",
                    )
                )
                continue

            actual_field = batch.schema.field(expected_field.name)

            # Check type
            if actual_field.type != expected_field.type:
                errors.append(
                    ValidationError(
                        field_name=expected_field.name,
                        error_type="type_mismatch",
                        message=(
                            f"Field '{expected_field.name}' has type {actual_field.type}, "
                            f"expected {expected_field.type}"
                        ),
                    )
                )

            # Check nullability
            if not expected_field.nullable:
                col = batch.column(expected_field.name)
                null_count = col.null_count
                if null_count > 0:
                    errors.append(
                        ValidationError(
                            field_name=expected_field.name,
                            error_type="null_violation",
                            message=(
                                f"Non-nullable field '{expected_field.name}' "
                                f"contains {null_count} null(s)"
                            ),
                        )
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            rows_validated=batch.num_rows,
        )


@dataclass
class SequenceGap:
    """Represents a detected gap in sequence numbers."""

    symbol: str
    expected: int
    actual: int
    gap_size: int


class SequenceTracker:
    """Tracks per-symbol sequence numbers and detects gaps.

    Useful for detecting dropped messages in market data feeds.
    """

    def __init__(self) -> None:
        self._sequences: dict[str, int] = {}
        self._gaps: list[SequenceGap] = []
        self._total_messages: int = 0
        self._total_gaps: int = 0

    @property
    def gaps(self) -> list[SequenceGap]:
        return list(self._gaps)

    @property
    def total_messages(self) -> int:
        return self._total_messages

    @property
    def total_gaps(self) -> int:
        return self._total_gaps

    def track(self, symbol: str, sequence: int) -> SequenceGap | None:
        """Track a sequence number and detect gaps.

        Args:
            symbol: The symbol/instrument identifier.
            sequence: The sequence number from the feed.

        Returns:
            A SequenceGap if a gap was detected, None otherwise.
        """
        self._total_messages += 1

        if symbol not in self._sequences:
            self._sequences[symbol] = sequence
            return None

        expected = self._sequences[symbol] + 1
        self._sequences[symbol] = sequence

        if sequence != expected and sequence > expected:
            gap = SequenceGap(
                symbol=symbol,
                expected=expected,
                actual=sequence,
                gap_size=sequence - expected,
            )
            self._gaps.append(gap)
            self._total_gaps += 1
            return gap

        return None

    def track_batch(self, batch: pa.RecordBatch) -> list[SequenceGap]:
        """Track sequences for an entire batch.

        Expects 'symbol' and 'sequence' columns in the batch.

        Args:
            batch: RecordBatch with symbol and sequence columns.

        Returns:
            List of detected gaps.
        """
        if "symbol" not in batch.schema.names or "sequence" not in batch.schema.names:
            return []

        symbols = batch.column("symbol").to_pylist()
        sequences = batch.column("sequence").to_pylist()
        gaps = []

        for sym, seq in zip(symbols, sequences):
            if seq is not None:
                gap = self.track(sym, seq)
                if gap is not None:
                    gaps.append(gap)

        return gaps

    def reset(self, symbol: str | None = None) -> None:
        """Reset tracking state.

        Args:
            symbol: If provided, reset only that symbol. Otherwise reset all.
        """
        if symbol is not None:
            self._sequences.pop(symbol, None)
        else:
            self._sequences.clear()
            self._gaps.clear()
            self._total_messages = 0
            self._total_gaps = 0
