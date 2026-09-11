#!/usr/bin/env python
"""Unit tests for LongLivedCredentials: a caller is never handed credentials below
``min_lifetime_seconds``, and the credential chain is walked once however many callers
arrive. Fake botocore session, no AWS.
"""

import datetime as dt
import unittest
from concurrent import futures
from unittest import mock

from agentcore_rl_toolkit.aws_tools.boto3_tools import (
    DEFAULT_MIN_LIFETIME_SECONDS,
    DEFAULT_SESSION_DURATION_SECONDS,
    LongLivedCredentials,
)

MODULE = "agentcore_rl_toolkit.aws_tools.boto3_tools"
ROLE_ARN = "arn:aws:iam::123456789012:role/TrainingHost"
ASSUMED_ARN = "arn:aws:sts::123456789012:assumed-role/TrainingHost/i-0abc"
USER_ARN = "arn:aws:iam::123456789012:user/danylo"


def in_hours(hours: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)


class FakeCredentials:
    """Botocore's ambient Credentials. ``_expiry_time`` is the private attribute
    ``expires_in`` falls back to when STS stated no expiry."""

    def __init__(self, token: str = "ambient", expiry: dt.datetime | None = None):
        self.access_key, self.secret_key, self.token = "AK-ambient", "SK", token
        self._expiry_time = expiry


class FakeSTS:
    def __init__(self, arn: str, assume_error: Exception | None = None):
        self.arn = arn
        self.assume_error = assume_error
        self.assume_calls: list[dict] = []
        self.identity_calls = 0
        self._sessions = 0

    def get_caller_identity(self):
        self.identity_calls += 1
        return {"Arn": self.arn}

    def assume_role(self, **kwargs):
        self.assume_calls.append(kwargs)
        if self.assume_error is not None:
            raise self.assume_error
        self._sessions += 1
        duration = kwargs["DurationSeconds"]
        return {
            "Credentials": {
                "AccessKeyId": f"AK-{self._sessions}",
                "SecretAccessKey": f"SK-{self._sessions}",
                "SessionToken": f"token-{self._sessions}",
                "Expiration": dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=duration),
            }
        }


class FakeSession:
    """A botocore Session whose chain walks are counted."""

    def __init__(self, sts: FakeSTS, ambient: FakeCredentials | None):
        self.sts = sts
        self.ambient = ambient
        self.chain_walks = 0

    def get_credentials(self):
        self.chain_walks += 1
        return self.ambient

    def create_client(self, name: str):
        assert name == "sts", name
        return self.sts


def patched(sts: FakeSTS, ambient: FakeCredentials | None = None):
    """Patch the Session the class constructs; returns (patch, fake session)."""
    session = FakeSession(sts, ambient if ambient is not None else FakeCredentials())
    return mock.patch("botocore.session.Session", return_value=session), session


class DefaultsTest(unittest.TestCase):
    def test_the_floor_is_below_the_session_duration(self):
        # Otherwise every load() would re-assume, defeating the caching.
        self.assertLess(DEFAULT_MIN_LIFETIME_SECONDS, DEFAULT_SESSION_DURATION_SECONDS)

    def test_the_duration_is_the_sts_maximum(self):
        self.assertEqual(DEFAULT_SESSION_DURATION_SECONDS, 12 * 60 * 60)


class AssumeRoleTest(unittest.TestCase):
    def test_the_ambient_role_is_assumed_with_an_explicit_duration(self):
        sts = FakeSTS(ASSUMED_ARN)
        patch, session = patched(sts)
        with patch:
            creds = LongLivedCredentials(duration_seconds=3600, role_session_name="unit-test")
            creds.load()

        # DurationSeconds is what makes the expiry known rather than inferred.
        self.assertEqual(
            sts.assume_calls,
            [
                {
                    "RoleArn": ROLE_ARN,
                    "RoleSessionName": "unit-test",
                    "DurationSeconds": 3600,
                }
            ],
        )
        self.assertEqual(session.chain_walks, 1)

    def test_the_assumed_role_arn_is_derived_from_the_caller_identity(self):
        # arn:aws:sts::<acct>:assumed-role/<role>/<session> -> the role's IAM arn.
        sts = FakeSTS(ASSUMED_ARN)
        patch, _ = patched(sts)
        with patch:
            LongLivedCredentials().load()
        self.assertEqual(sts.assume_calls[0]["RoleArn"], ROLE_ARN)

    def test_load_returns_the_assumed_session_not_the_ambient_one(self):
        sts = FakeSTS(ASSUMED_ARN)
        patch, _ = patched(sts, FakeCredentials(token="ambient"))
        with patch:
            got = LongLivedCredentials().load()
        self.assertEqual(got.token, "token-1")

    def test_expires_in_reports_the_duration_sts_stated(self):
        sts = FakeSTS(ASSUMED_ARN)
        patch, _ = patched(sts)
        with patch:
            creds = LongLivedCredentials(duration_seconds=8 * 3600)
            creds.load()
            remaining = creds.expires_in()
        self.assertAlmostEqual(remaining, 8 * 3600, delta=5)

    def test_credentials_above_the_floor_are_reused(self):
        sts = FakeSTS(ASSUMED_ARN)
        patch, session = patched(sts)
        with patch:
            creds = LongLivedCredentials(duration_seconds=12 * 3600, min_lifetime_seconds=6 * 3600)
            tokens = {creds.load().token for _ in range(50)}
        self.assertEqual(tokens, {"token-1"})
        self.assertEqual(len(sts.assume_calls), 1)
        self.assertEqual(session.chain_walks, 1)

    def test_credentials_below_the_floor_are_re_assumed(self):
        # A floor above the duration makes every set stale on arrival.
        sts = FakeSTS(ASSUMED_ARN)
        patch, session = patched(sts)
        with patch:
            creds = LongLivedCredentials(duration_seconds=3600, min_lifetime_seconds=99999)
            tokens = [creds.load().token for _ in range(3)]
        self.assertEqual(tokens, ["token-1", "token-2", "token-3"])
        # Re-assuming must not re-walk the chain: the session already owns it.
        self.assertEqual(session.chain_walks, 1)

    def test_every_load_is_above_the_floor(self):
        sts = FakeSTS(ASSUMED_ARN)
        patch, _ = patched(sts)
        with patch:
            creds = LongLivedCredentials(duration_seconds=3600, min_lifetime_seconds=1800)
            for _ in range(5):
                creds.load()
                self.assertGreaterEqual(creds.expires_in(), creds.min_lifetime_seconds)


class AmbientFallbackTest(unittest.TestCase):
    def test_a_non_role_caller_keeps_the_ambient_credentials(self):
        # An IAM user has no role to assume, so it works without a floor.
        sts = FakeSTS(USER_ARN)
        ambient = FakeCredentials(token="ambient", expiry=in_hours(1))
        patch, _ = patched(sts, ambient)
        with patch:
            creds = LongLivedCredentials()
            got = creds.load()
        self.assertIs(got, ambient)
        self.assertEqual(sts.assume_calls, [])

    def test_a_refused_assume_role_falls_back_and_warns(self):
        # The trust policy may not let the role assume itself; degrade, do not fail.
        sts = FakeSTS(ASSUMED_ARN, assume_error=RuntimeError("AccessDenied"))
        ambient = FakeCredentials(token="ambient", expiry=in_hours(1))
        patch, _ = patched(sts, ambient)
        with patch, self.assertLogs(MODULE, level="WARNING") as logs:
            creds = LongLivedCredentials()
            got = creds.load()
        self.assertIs(got, ambient)
        self.assertIn("no floor", "".join(logs.output))

    def test_the_ambient_fallback_never_retries_the_assume(self):
        # _role_arn is cleared on the way into the fallback.
        sts = FakeSTS(ASSUMED_ARN, assume_error=RuntimeError("AccessDenied"))
        patch, _ = patched(sts, FakeCredentials(expiry=in_hours(0.1)))
        with patch, self.assertLogs(MODULE, level="WARNING"):
            creds = LongLivedCredentials()
            for _ in range(5):
                creds.load()
        self.assertEqual(len(sts.assume_calls), 1)

    def test_expires_in_falls_back_to_botocores_optimistic_expiry(self):
        sts = FakeSTS(USER_ARN)
        patch, _ = patched(sts, FakeCredentials(expiry=in_hours(2)))
        with patch:
            creds = LongLivedCredentials()
            creds.load()
            self.assertAlmostEqual(creds.expires_in(), 2 * 3600, delta=5)

    def test_expires_in_is_none_when_nothing_states_an_expiry(self):
        # Long-lived static keys: unknown is reported as unknown, not as a number.
        sts = FakeSTS(USER_ARN)
        patch, _ = patched(sts, FakeCredentials(expiry=None))
        with patch:
            creds = LongLivedCredentials()
            creds.load()
            self.assertIsNone(creds.expires_in())

    def test_ambient_mode_never_re_assumes_however_low_the_floor_looks(self):
        # Nothing to re-assume to, so the floor check is gated on _role_arn rather
        # than on the expiry.
        for expiry in (None, in_hours(0.01)):
            sts = FakeSTS(USER_ARN)
            patch, _ = patched(sts, FakeCredentials(expiry=expiry))
            with patch:
                creds = LongLivedCredentials(min_lifetime_seconds=99999)
                for _ in range(5):
                    creds.load()
            self.assertEqual(sts.assume_calls, [], f"re-assumed with expiry={expiry}")

    def test_no_credentials_at_all_raises(self):
        sts = FakeSTS(USER_ARN)
        patch = mock.patch("botocore.session.Session", return_value=FakeSession(sts, None))
        with patch:
            with self.assertRaises(RuntimeError):
                LongLivedCredentials().load()


class ThreadSafetyTest(unittest.TestCase):
    def test_a_concurrent_first_touch_walks_the_chain_once(self):
        # Resolving the chain is the rate-limited part (IMDS), so a burst of
        # first-touches must collapse into one walk and one session.
        sts = FakeSTS(ASSUMED_ARN)
        patch, session = patched(sts)
        with patch:
            creds = LongLivedCredentials()
            with futures.ThreadPoolExecutor(max_workers=16) as pool:
                tokens = set(pool.map(lambda _: creds.load().token, range(64)))

        self.assertEqual(session.chain_walks, 1)
        self.assertEqual(len(sts.assume_calls), 1)
        self.assertEqual(tokens, {"token-1"})

    def test_load_is_the_whole_provider_interface(self):
        # aws_bedrock_token_generator takes this as an `aws_credentials_provider`
        # and only ever calls load().
        self.assertTrue(callable(LongLivedCredentials().load))


if __name__ == "__main__":
    unittest.main()
