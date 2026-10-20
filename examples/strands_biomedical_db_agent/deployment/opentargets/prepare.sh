#!/usr/bin/env bash
# Download and extract the pinned public release into an unused data directory.
set -euo pipefail
: "${OT_DATA_ROOT:?Set OT_DATA_ROOT to an unused absolute data directory}"
case "$OT_DATA_ROOT" in
    /*) ;;
    *) echo "OT_DATA_ROOT must be absolute" >&2; exit 1 ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/release-26.06.env"
if [[ -e "$OT_DATA_ROOT" ]]; then
    if [[ -f "$OT_DATA_ROOT/.release" ]] && [[ "$(<"$OT_DATA_ROOT/.release")" == "$OT_RELEASE" ]]; then
        echo "Release $OT_RELEASE is already prepared in $OT_DATA_ROOT"
        exit 0
    fi
    echo "Refusing to extract into an existing directory: $OT_DATA_ROOT" >&2
    exit 1
fi
for command in curl tar pigz sha1sum; do
    command -v "$command" >/dev/null
done
scratch=$(mktemp -d -t opentargets.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p -- "$(dirname -- "$OT_DATA_ROOT")"
mkdir -- "$OT_DATA_ROOT"
for database in clickhouse opensearch; do
    archive="$scratch/$database.tgz"
    curl --fail --location --retry 3 \
        "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/$OT_RELEASE/disk_images/$database.tgz" \
        --output "$archive"
    if [[ "$database" == clickhouse ]]; then
        checksum="$OT_CLICKHOUSE_SHA1"
    else
        checksum="$OT_OPENSEARCH_SHA1"
    fi
    printf '%s  %s\n' "$checksum" "$archive" | sha1sum --check
    mkdir -- "$OT_DATA_ROOT/$database"
    tar --use-compress-program=pigz --no-same-owner -xf "$archive" -C "$OT_DATA_ROOT/$database"
    rm -- "$archive"
done
printf '%s\n' "$OT_RELEASE" > "$OT_DATA_ROOT/.release"
echo "Prepared Open Targets $OT_RELEASE in $OT_DATA_ROOT"
