"""Download verified public GWAS files and optionally publish them to your S3 prefix."""

import argparse
import base64
import shutil
import tempfile
import urllib.request
from pathlib import Path

from botocore.exceptions import ClientError

from .catalog import DATASETS, DEFAULT_DATA_ROOT
from .hashing import sha256_file
from .s3_snapshot import object_args, s3_client, verify_objects


def verify(path, dataset):
    if path.stat().st_size != dataset.size_bytes or sha256_file(path) != dataset.sha256:
        raise ValueError(f"{path}: size/SHA-256 mismatch; remove this file before retrying")


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gwas-download-") as scratch:
        for dataset in DATASETS:
            target = directory / dataset.filename
            if not target.exists():
                temporary = Path(scratch) / dataset.filename
                with (
                    urllib.request.urlopen(dataset.source_url, timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
                verify(temporary, dataset)
                shutil.copyfile(temporary, target)
                temporary.unlink()
            verify(target, dataset)
            print(f"Verified {target}")


def publish(directory: Path, prefix: str) -> None:
    client = s3_client()
    sources = {dataset.name: f"{prefix.rstrip('/')}/{dataset.filename}" for dataset in DATASETS}
    for dataset in DATASETS:
        path = directory / dataset.filename
        verify(path, dataset)
        args = object_args(sources[dataset.name])
        try:
            client.head_object(**args)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
            with path.open("rb") as body:
                client.put_object(
                    **args,
                    Body=body,
                    IfNoneMatch="*",
                    ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=base64.b64encode(bytes.fromhex(dataset.sha256)).decode(),
                )
    # Existing objects must match; never overwrite an existing release.
    verify_objects(sources, client=client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--s3-prefix", help="Optional immutable s3://bucket/release-prefix")
    args = parser.parse_args()
    if args.s3_prefix:
        object_args(f"{args.s3_prefix.rstrip('/')}/check")
    download(args.directory)
    if args.s3_prefix:
        publish(args.directory, args.s3_prefix)


if __name__ == "__main__":
    main()
