import os
import shutil
import tempfile

from migration_bench.common import eval_utils, hash_utils, maven_utils, utils
from migration_bench.lang.java.eval.parse_repo import same_repo_test_files

from agentcore_rl_toolkit import RewardFunction


class MigrationReward(RewardFunction):
    def __call__(
        self,
        repo_dir: str,
        original_num_tests: int,
        original_commit_id: str,
        require_maximal_migration: bool = False,
        **kwargs,
    ):
        """
        There are two main criteria for migration:
        - build success with java 17
        - test cases are preserved, both in terms of quantity and functional equivalence

        when the build is not successful, we directly return a reward of zero, as the repo
        at its original commit will pass the `eval_test_equivalence` without any migration changes.
        """

        reward = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            # copy repo_dir to a temp dir for evaluation
            temp_repo_dir = os.path.join(temp_dir, os.path.basename(repo_dir))
            shutil.copytree(repo_dir, temp_repo_dir)

            if self.eval_build_success(repo_dir=temp_repo_dir, require_maximal_migration=require_maximal_migration):
                print("build succeeded!")
                reward += 0.5
            else:
                print("build failed!")
                return reward

            if self.eval_test_equivalence(
                repo_dir=temp_repo_dir,
                original_num_tests=original_num_tests,
                original_commit_id=original_commit_id,
            ):
                print("test equivalence check passed!")
                reward += 0.5
            else:
                print("test equivalence check failed!")

        return reward

    @staticmethod
    def eval_build_success(
        repo_dir: str,
        require_compiled_java_major_version: int = 61,
        maven_command: str = maven_utils.MVN_CLEAN_VERIFY,
        require_maximal_migration: bool = False,
    ):
        build_success = (
            (maven_utils.do_run_maven_command(maven_command.format(root_dir=repo_dir), check=False).return_code == 0)
            and (
                (require_compiled_java_major_version is None)
                or (utils.get_compiled_java_major_versions(repo_dir) == {require_compiled_java_major_version})
            )
            and ((not require_maximal_migration) or eval_utils.check_version(repo_dir))
        )
        return build_success

    @staticmethod
    def eval_test_equivalence(repo_dir: str, original_num_tests: int, original_commit_id: str):
        mvn_tests = maven_utils.do_run_maven_command(maven_utils.MVN_NUM_TESTS.format(root_dir=repo_dir), check=False)
        num_tests = hash_utils.get_num_test_cases(repo_dir, mvn_tests.stdout)

        num_tests_match = num_tests is not None and num_tests >= original_num_tests
        func_tests_match = same_repo_test_files(repo_dir, lhs_branch=original_commit_id)[-1]
        return num_tests_match and func_tests_match


if __name__ == "__main__":
    reward_fn = MigrationReward()
    reward = reward_fn(
        repo_dir="/tmp/workspace/EJServer",
        original_commit_id="7e51c59e090484ae4573290099b6936855554064",
        original_num_tests=1,
        require_maximal_migration=False,
    )
    print(reward)
