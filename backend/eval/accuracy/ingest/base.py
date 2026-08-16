"""Base interface for historical-data sources in the accuracy pipeline.

Every ingest adapter (AQS daily summaries now; FIRMS-historical, HMS, NIFC,
and weather in later chunks) implements ``ArchiveSource`` so the pipeline can
ingest any source uniformly. Keep the contract minimal: a source yields
canonical records for a year, with adapter-specific knobs passed through as
keyword arguments (e.g. a bounding box or region filter).
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar

T = TypeVar("T")


class ArchiveSource(ABC, Generic[T]):
    """A source of historical ground-truth records for a given year.

    Subclasses declare ``source_name`` and implement :meth:`fetch`, yielding
    the source's canonical record type (e.g. ``AqsDailyRecord``). Records must
    be suitable for ``AccuracyStore`` persistence.
    """

    source_name: str = "archive"

    @abstractmethod
    def fetch(self, year: int, **kwargs: Any) -> Iterable[T]:
        """Yield canonical records published during ``year``.

        Adapter-specific options (bounding box, region, parameters, ...) are
        passed as keyword arguments.
        """
        raise NotImplementedError
