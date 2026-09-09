"""boto3/aioboto3 plumbing: cached sessions, self-assumed role credentials with a known
lifetime, and small shared helpers.
"""

import datetime as dt
import logging
import threading
from asyncio import Lock
from functools import lru_cache
from threading import local

import aioboto3
import boto3

logger = logging.getLogger(__name__)

thread_local = local()

_aioboto3_lock = Lock()
_aioboto3_credentials_ready = False


def get_boto3_session() -> boto3.Session:
    if "boto3_session" not in thread_local.__dict__:
        thread_local.boto3_session = boto3.Session()
    return thread_local.boto3_session


@lru_cache
def _aioboto3_session() -> aioboto3.Session:
    return aioboto3.Session()


async def get_aioboto3_session() -> aioboto3.Session:
    """Return the process-wide aioboto3 session with credentials resolved.

    ``AioSession.get_credentials()`` resolves lazily and without a lock, so a concurrent
    burst would all walk the provider chain in parallel and IMDS starts failing after a
    hundred or so calls. Resolve once behind our own lock instead.
    """
    global _aioboto3_credentials_ready
    session = _aioboto3_session()
    if not _aioboto3_credentials_ready:
        async with _aioboto3_lock:
            if not _aioboto3_credentials_ready:
                await session.get_credentials()
                _aioboto3_credentials_ready = True
    return session


def get_role_credentials(role_arn: str) -> dict[str, str]:
    """Temporary credentials that can only be used for short tasks (<1hr)."""
    sts = get_boto3_session().client("sts")

    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="agentcore_rl_toolkit",
    )

    creds = response["Credentials"]

    return {
        "AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"],
        "AWS_SESSION_TOKEN": creds["SessionToken"],
    }


# --- credentials with a known lifetime ----------------------------------------
#
# Ambient credentials promise nothing about their remaining lifetime, and botocore's
# reported expiry is optimistic (a degraded IMDS pushes it past the true expiration so it
# can retry). Callers that must commit to a lifetime up front -- presigning a URL, minting
# a token -- need a floor, and only `AssumeRole` gives one, via an explicit
# DurationSeconds. 12h is reachable because EC2 instance-profile credentials are exempt
# from the one-hour role-chaining cap.

DEFAULT_SESSION_DURATION_SECONDS = 12 * 60 * 60  # STS max for a self-assumed role
DEFAULT_MIN_LIFETIME_SECONDS = 6 * 60 * 60  # re-assume below this


class LongLivedCredentials:
    """Shared, thread-safe credentials with a known floor on their remaining lifetime.

    :meth:`load` returns an STS session for the ambient role, re-assumed once it falls
    below ``min_lifetime_seconds``. Resolved once as process state, so callers can sign
    cheaply per use instead of caching signatures that may outlive their work.

    ``load`` is the whole botocore credentials-provider interface, so an instance can be
    passed anywhere one is expected. If the role cannot assume itself (its trust policy
    must allow it) this falls back to the ambient chain and the floor is lost;
    :meth:`expires_in` reports what is known either way.
    """

    def __init__(
        self,
        duration_seconds: int = DEFAULT_SESSION_DURATION_SECONDS,
        min_lifetime_seconds: int = DEFAULT_MIN_LIFETIME_SECONDS,
        role_session_name: str = "agentcore_rl_toolkit",
    ) -> None:
        self.duration_seconds = duration_seconds
        self.min_lifetime_seconds = min_lifetime_seconds
        self.role_session_name = role_session_name
        self._lock = threading.Lock()
        self._session = None
        self._credentials = None
        self._expires_at: dt.datetime | None = None
        self._role_arn: str | None = None
        self._initialized = False

    def load(self):
        """The current credentials, re-assuming first if they are below the floor."""
        with self._lock:
            if not self._initialized:
                self._initialize()
            elif self._role_arn is not None and self._expires_in() < self.min_lifetime_seconds:
                self._assume_role()
            return self._credentials

    def expires_in(self) -> float | None:
        """Seconds of life left, or ``None`` if unknown.

        Exact in assume-role mode; in the ambient fallback it is botocore's optimistic
        expiry -- good enough for a warning, not for concluding a signature is safe.
        """
        expires_at = self._expires_at or getattr(self._credentials, "_expiry_time", None)
        if expires_at is None:
            return None
        return (expires_at - dt.datetime.now(dt.timezone.utc)).total_seconds()

    def _expires_in(self) -> float:
        """:meth:`expires_in` with "never expires" folded in as infinity."""
        remaining = self.expires_in()
        return float("inf") if remaining is None else remaining

    def _initialize(self) -> None:
        from botocore.session import Session

        # One Session for the whole process rather than the thread-local
        # get_boto3_session(), so the credential chain is walked exactly once.
        self._session = Session()
        logger.info("resolving AWS credentials")
        ambient = self._session.get_credentials()
        if ambient is None:
            raise RuntimeError("No AWS credentials found.")
        self._initialized = True

        self._role_arn = self._caller_role_arn()
        if self._role_arn is not None:
            try:
                self._assume_role()
                return
            except Exception as error:
                logger.warning(
                    "cannot assume %s for a long-lived session (%s); falling back to ambient "
                    "credentials, whose remaining lifetime has no floor",
                    self._role_arn,
                    error,
                )
                self._role_arn = None
        self._credentials = ambient
        self._expires_at = None

    def _caller_role_arn(self) -> str | None:
        """The IAM role behind the ambient credentials, or ``None`` if not a role session."""
        arn = self._session.create_client("sts").get_caller_identity()["Arn"]
        # arn:aws:sts::<acct>:assumed-role/<role>/<session> -> arn:aws:iam::<acct>:role/<role>
        fields = arn.split(":")
        resource = fields[-1].split("/")
        if resource[0] != "assumed-role":
            return None
        return f"arn:aws:iam::{fields[4]}:role/{resource[1]}"

    def _assume_role(self) -> None:
        from botocore.credentials import Credentials

        assumed = self._session.create_client("sts").assume_role(
            RoleArn=self._role_arn,
            RoleSessionName=self.role_session_name,
            DurationSeconds=self.duration_seconds,
        )["Credentials"]
        self._credentials = Credentials(assumed["AccessKeyId"], assumed["SecretAccessKey"], assumed["SessionToken"])
        self._expires_at = assumed["Expiration"]
        logger.info("assumed %s; session expires at %s", self._role_arn, self._expires_at)


def tags_to_map(tags):
    d = {}
    for line in tags:
        d[line["Key"]] = line["Value"]
    return d
