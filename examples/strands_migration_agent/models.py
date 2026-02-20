from pydantic import BaseModel


class InvocationRequest(BaseModel):
    prompt: str
    repo_uri: str
    metadata_uri: str
    require_maximal_migration: bool


class RepoMetaData(BaseModel):
    repo: str
    base_commit: str
    num_java_files: int
    num_loc: int
    num_pom_xml: int
    num_src_test_java_files: int
    num_test_cases: int
    license: str
