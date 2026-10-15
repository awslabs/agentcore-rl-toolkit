import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from biomedical_db_agent.gwas_catalog import prepare, s3_snapshot


def test_download_checks_bytes_before_install(tmp_path, monkeypatch):
    import io

    body = b"public test bytes"
    dataset = SimpleNamespace(
        name="fixture",
        filename="fixture.parquet",
        source_url="https://example.test/fixture",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )
    monkeypatch.setattr(prepare, "DATASETS", (dataset,))
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(body))
    directory = tmp_path / "download"
    prepare.download(directory)
    assert (directory / dataset.filename).read_bytes() == body
    monkeypatch.setattr(prepare.urllib.request, "urlopen", Mock(side_effect=AssertionError("must reuse verified file")))
    prepare.download(directory)
    (directory / dataset.filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        prepare.download(directory)


def test_s3_publish_does_not_overwrite_and_checks_full_hashes(tmp_path, monkeypatch):
    data = b"synthetic parquet placeholder"
    dataset = SimpleNamespace(
        name="fixture",
        filename="fixture.parquet",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    (tmp_path / dataset.filename).write_bytes(data)
    client = Mock()
    not_found = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    checksum = base64.b64encode(bytes.fromhex(dataset.sha256)).decode()
    head = {"ContentLength": len(data), "ChecksumSHA256": checksum, "ChecksumType": "FULL_OBJECT"}
    client.head_object.side_effect = [not_found, head]
    monkeypatch.setattr(prepare, "s3_client", lambda: client)
    monkeypatch.setattr(prepare, "DATASETS", (dataset,))
    monkeypatch.setattr(s3_snapshot, "DATASETS", (dataset,))
    prepare.publish(tmp_path, "s3://example-bucket/release")
    assert client.put_object.call_args.kwargs["IfNoneMatch"] == "*"
    assert client.put_object.call_args.kwargs["ChecksumSHA256"] == checksum
    client.reset_mock()
    client.head_object.side_effect = [head, {**head, "ChecksumSHA256": "wrong"}]
    with pytest.raises(ValueError, match="SHA-256 differs"):
        prepare.publish(tmp_path, "s3://example-bucket/release")
    client.put_object.assert_not_called()
