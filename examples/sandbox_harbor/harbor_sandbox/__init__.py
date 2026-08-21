"""Harbor-on-AgentCore glue: naming convention + sandbox lifecycle + image build.

Built on the toolkit's generic ``SandboxClient`` (session API), adding the
Harbor-benchmark-specific half: one naming convention (``resolve``), a client
that creates/releases a task's runtime on demand (``HarborSandboxClient``), a
builder that turns a Harbor dataset into runnable images (``build``), and a
puller that fetches a benchmark's tasks from the Harbor registry
(``ensure_dataset``). Deployment constants live in ``config``; shared exceptions
in ``errors``.
"""

from .client import HarborSandboxClient
from .config import REGION, S3_BUCKET
from .dataset import ensure_dataset
from .errors import ImageNotFoundError, ValidationError
from .naming import SandboxNames, resolve

__all__ = [
    "resolve",
    "SandboxNames",
    "ensure_dataset",
    "ValidationError",
    "REGION",
    "S3_BUCKET",
    "HarborSandboxClient",
    "ImageNotFoundError",
]
