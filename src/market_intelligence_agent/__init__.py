"""Market Intelligence Agent - autonomous, source-grounded competitor research."""

from .config import Settings
from .models import AgentResult, Brief, BriefSection, SearchPlan, SourceRecord, SubQuestion

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "Brief",
    "BriefSection",
    "SearchPlan",
    "Settings",
    "SourceRecord",
    "SubQuestion",
    "__version__",
]
