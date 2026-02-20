import argparse
import json
import os
import subprocess
import tarfile

import pandas as pd


def check_bucket_exists(s3_bucket_name: str) -> bool:
    """Check if S3 bucket exists and is accessible."""
    try:
        subprocess.run(
            ["aws", "s3", "ls", f"s3://{s3_bucket_name}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main(args):
    # Check S3 bucket exists before processing (unless skipping sync)
    if not args.skip_s3_sync:
        print(f"Checking S3 bucket: {args.s3_bucket_name}")
        if not check_bucket_exists(args.s3_bucket_name):
            print(f"ERROR: S3 bucket '{args.s3_bucket_name}' does not exist or is not accessible.")
            print(f"Create it with: aws s3 mb s3://{args.s3_bucket_name}")
            return

    print("Loading datasets...")
    df_full = pd.read_csv("hf://datasets/AmazonScience/migration-bench-java-full/test.csv")
    df_test = pd.read_csv("hf://datasets/AmazonScience/migration-bench-java-selected/test.csv")

    df_train = create_train_data(df_full, df_test)

    # Apply limit if specified (useful for quick testing)
    if args.max_repos_per_split:
        df_train = df_train.head(args.max_repos_per_split)
        df_test = df_test.head(args.max_repos_per_split)

    print(f"Train set: {len(df_train)} repos")
    print(f"Test set: {len(df_test)} repos")

    # Process train repositories
    train_summary = process_repos(df_train, "train", args.workspace_dir, args.s3_bucket_name)

    # Process test repositories
    test_summary = process_repos(df_test, "test", args.workspace_dir, args.s3_bucket_name)

    # Print summary before sync
    print_summary(train_summary, test_summary, args.workspace_dir, args.s3_bucket_name)

    # Sync to S3
    if args.skip_s3_sync:
        print("Skipping S3 sync (--skip-s3-sync flag set)")
        print("All done!")
    else:
        sync_success = sync_to_s3(args.workspace_dir, args.s3_bucket_name)
        if sync_success:
            print("All done!")
        else:
            print("Processing complete, but S3 sync failed. Check errors above.")


def print_summary(train_summary: dict, test_summary: dict, workspace_dir: str, s3_bucket_name: str):
    """Print and save final processing summary."""
    all_failed = train_summary["failed_repos"] + test_summary["failed_repos"]
    total_successful = train_summary["successful"] + test_summary["successful"]
    total_failed = train_summary["failed"] + test_summary["failed"]
    total_repos = total_successful + total_failed

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(
        f'Train: {train_summary["successful"]}/{train_summary["successful"] + train_summary["failed"]} successful, '
        f'{train_summary["failed"]} failed'
    )
    print(
        f'Test: {test_summary["successful"]}/{test_summary["successful"] + test_summary["failed"]} successful, '
        f'{test_summary["failed"]} failed'
    )
    print(f"Total: {total_successful}/{total_repos} successful, {total_failed}/{total_repos} failed")

    if all_failed:
        print(f"Failed repositories ({len(all_failed)}):")
        for failed in all_failed:
            print(f'  - {failed["repo"]} ({failed["split"]})')
            print(f'    Error: {failed["error"]}')

        # Save failed repos to JSON
        summary_dir = os.path.join(workspace_dir, s3_bucket_name)
        failed_repos_path = os.path.join(summary_dir, "failed_repos.json")
        with open(failed_repos_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total_failed": total_failed,
                        "train_failed": train_summary["failed"],
                        "test_failed": test_summary["failed"],
                    },
                    "failed_repos": all_failed,
                },
                f,
                indent=2,
            )
        print(f"Failed repos saved to: {failed_repos_path}")

    print("=" * 60)


def create_train_data(df_full: pd.DataFrame, df_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Create training data by removing selected repos from the full dataset.

    Args:
        df_full: The complete dataset
        df_selected: The subset to exclude (test set)

    Returns:
        DataFrame: df_full - df_selected (training set)
    """
    # Get repos that are in df_selected
    selected_repos = df_selected["repo"].unique()

    # Filter out rows with repos that are in df_selected
    train_df = df_full[~df_full["repo"].isin(selected_repos)]

    return train_df


def setup_repo(repo_metadata: pd.Series, workspace_dir: str):
    repo_id = repo_metadata.repo
    base_commit = repo_metadata.base_commit

    # Create workspace directory if it doesn't exist
    os.makedirs(workspace_dir, exist_ok=True)

    # Use {repo_author}__{repo_name} format for directory name
    repo_author, repo_name = repo_id.split("/")
    repo_dir_name = f"{repo_author}__{repo_name}"
    repo_path = os.path.join(workspace_dir, repo_dir_name)

    # Remove existing directory if it exists
    if os.path.exists(repo_path):
        subprocess.run(["rm", "-rf", repo_path], check=True)

    # Clone the repository
    repo_url = f"https://github.com/{repo_id}.git"
    subprocess.run(["git", "clone", repo_url, repo_path], check=True, capture_output=True)

    # Change to repo directory and checkout base commit
    subprocess.run(["git", "checkout", base_commit], cwd=repo_path, check=True, capture_output=True)

    return repo_path


def save_metadata(repo_metadata: pd.Series, output_dir: str):
    """
    Save repository metadata to metadata.json.

    Args:
        repo_metadata: DataFrame row containing repo metadata
        output_dir: Directory to save metadata.json
    """
    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata = repo_metadata.to_dict()
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def tar_repo(repo_metadata: pd.Series, repo_path: str, output_dir: str):
    """
    Create tar.gz file for a repository.

    Args:
        repo_metadata: DataFrame row containing repo metadata
        repo_path: Path to the cloned repository
        output_dir: Directory to save tar file
    """
    repo_author, repo_name = repo_metadata.repo.split("/")
    repo_dir_name = f"{repo_author}__{repo_name}"

    # Verify repo_path matches expected format
    if not repo_path.endswith(repo_dir_name):
        raise ValueError(f"Repo path mismatch: expected path ending with {repo_dir_name}, got {repo_path}")

    # Create output directory for this specific repo
    tar_output_dir = os.path.join(output_dir, repo_dir_name)
    os.makedirs(tar_output_dir, exist_ok=True)

    # Create tar.gz file
    tar_file_path = os.path.join(tar_output_dir, f"{repo_dir_name}.tar.gz")
    with tarfile.open(tar_file_path, "w:gz") as tar:
        tar.add(repo_path, arcname=repo_dir_name)

    return tar_file_path


def process_repos(df: pd.DataFrame, split_name: str, workspace_dir: str, s3_bucket_name: str):
    """
    Process all repositories in a dataframe: clone, tar, and save metadata.

    Args:
        df: DataFrame containing repository metadata
        split_name: Either "train" or "test"
        workspace_dir: Workspace directory for cloning repos
        s3_bucket_name: S3 bucket name

    Returns:
        dict: Summary with 'successful', 'failed', and 'failed_repos' list
    """
    # Create directory structure:
    # - {workspace_dir}/{s3_bucket_name}/repos/{train|test}/ for uncompressed repos
    # - {workspace_dir}/{s3_bucket_name}/tar/{train|test}/ for tar files and metadata
    repos_dir = os.path.join(workspace_dir, s3_bucket_name, "repos", split_name)
    tars_dir = os.path.join(workspace_dir, s3_bucket_name, "tars", split_name)
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(tars_dir, exist_ok=True)

    print(f"Processing {len(df)} {split_name} repositories...")

    failed_repos = []
    successful_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  [{i + 1}/{len(df)}] {row.repo}...", end=" ", flush=True)

        try:
            repo_path = setup_repo(row, repos_dir)
            tar_path = tar_repo(row, repo_path, tars_dir)
            save_metadata(row, os.path.dirname(tar_path))
            print("done")
            successful_count += 1

        except Exception as e:
            error_msg = str(e)
            print(f"FAILED: {error_msg}")
            failed_repos.append({"repo": row.repo, "error": error_msg, "split": split_name})

    return {"successful": successful_count, "failed": len(failed_repos), "failed_repos": failed_repos}


def sync_to_s3(workspace_dir: str, s3_bucket_name: str):
    """
    Sync the prepared dataset to S3.

    Args:
        workspace_dir: Workspace directory containing the prepared data
        s3_bucket_name: S3 bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    local_path = os.path.join(workspace_dir, s3_bucket_name)
    s3_path = f"s3://{s3_bucket_name}/"

    print(f"Syncing to {s3_path}...", end=" ", flush=True)

    try:
        subprocess.run(["aws", "s3", "sync", local_path, s3_path], check=True, capture_output=True, text=True)
        print("done")
        return True

    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"  Error: {e.stderr}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MigrationBench dataset and upload to S3")
    parser.add_argument(
        "--workspace-dir", type=str, default="/tmp/workspace", help="Workspace directory for cloning repos"
    )
    parser.add_argument(
        "--max-repos-per-split", type=int, default=None, help="Max repos to process per split (for debugging)"
    )
    parser.add_argument(
        "--s3-bucket-name",
        type=str,
        required=True,
        help="S3 bucket name (must exist; create with: aws s3 mb s3://BUCKET_NAME)",
    )
    parser.add_argument("--skip-s3-sync", action="store_true", help="Skip S3 sync (for local testing)")
    args = parser.parse_args()

    main(args)
