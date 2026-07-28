from pydantic import BaseModel


class InvocationRequest(BaseModel):
    # `prompt` is typed `str` on purpose. It is consumed via `.format(...)` and passed
    # to an agent holding shell + editor tools, so pydantic rejecting any non-string
    # value (e.g. a list of content blocks) before the agent runs is a security control,
    # not just a convenience. Do NOT relax this to `str | list` / `Any` for multi-turn:
    # accepting a `toolUse` content block here would let a caller bypass model invocation
    # and dispatch a tool directly (Strands event-loop dispatch, H1-3679111).
    prompt: str
    repo_uri: str
    metadata_uri: str
    require_maximal_migration: bool
    use_dependency_search_tool: bool = False
    apply_static_update: bool = False


class RepoMetaData(BaseModel):
    repo: str
    base_commit: str
    num_java_files: int
    num_loc: int
    num_pom_xml: int
    num_src_test_java_files: int
    num_test_cases: int
    license: str
