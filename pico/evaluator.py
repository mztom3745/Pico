import hashlib
import json
import locale as locale_module
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import memory as memorylib
from .models import FakeModelClient
from .runtime import Pico, SessionStore
from .run_store import RunStore
from .task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from .workspace import WorkspaceContext

BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_PATH = Path("benchmarks/coding_tasks.json")
DEFAULT_ARTIFACT_PATH = Path("benchmarks/benchmark-v1.json")
DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH = Path("artifacts/harness-regression-v2.json")
DEFAULT_MODEL_NAME = "FakeModelClient"
DEFAULT_MODEL_VERSION = "scripted-deterministic"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TIMEZONE = "Asia/Shanghai"

REQUIRED_BENCHMARK_KEYS = ("schema_version", "tasks")
REQUIRED_TASK_KEYS = (
    "id",
    "prompt",
    "fixture_repo",
    "allowed_tools",
    "step_budget",
    "expected_artifact",
    "verifier",
    "category",
)

TASK_FIXTURE_ARTIFACTS = {
    "bench_repo_readme": "README.md",
    "bench_repo_patch": "sample.txt",
}

SCRIPTED_MODEL_OUTPUTS = {
    "readme_intro_locked": [
        '<tool name="patch_file" path="README.md"><old_text>This is a placeholder benchmark fixture.</old_text><new_text>This fixture is a locked benchmark workspace.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "readme_schema_note": [
        '<tool name="patch_file" path="README.md"><old_text>- Placeholder note about the repo.</old_text><new_text>- The benchmark schema and baseline are fixed.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "readme_ordering_note": [
        '<tool name="patch_file" path="README.md"><old_text>- Placeholder note about the file layout.</old_text><new_text>- Deterministic file ordering keeps benchmark diffs stable.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_beta_locked": [
        '<tool name="patch_file" path="sample.txt"><old_text>beta</old_text><new_text>beta-locked</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_gamma_locked": [
        '<tool name="patch_file" path="sample.txt"><old_text>gamma</old_text><new_text>gamma-locked</new_text></tool>',
        "<final>Done.</final>",
    ],
    "sample_placeholder_delta": [
        '<tool name="patch_file" path="sample.txt"><old_text>placeholder</old_text><new_text>delta</new_text></tool>',
        "<final>Done.</final>",
    ],
    "invalid_patch_recovery": [
        '<tool>{"name":"patch_file","args":{"path":"README.md","old_text":"This is a placeholder benchmark fixture."}}</tool>',
        '<tool name="patch_file" path="README.md"><old_text>This is a placeholder benchmark fixture.</old_text><new_text>This fixture recovered after invalid patch args.</new_text></tool>',
        "<final>Done.</final>",
    ],
    "path_escape_recovery": [
        '<tool>{"name":"read_file","args":{"path":"../outside.txt","start":1,"end":1}}</tool>',
        '<tool name="patch_file" path="sample.txt"><old_text>alpha</old_text><new_text>alpha-guarded</new_text></tool>',
        "<final>Done.</final>",
    ],
    "repeated_read_recovery": [
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool>{"name":"read_file","args":{"path":"sample.txt","start":1,"end":4}}</tool>',
        '<tool name="patch_file" path="sample.txt"><old_text>placeholder</old_text><new_text>repeat-guarded</new_text></tool>',
        "<final>Done.</final>",
    ],
    "context_reduction_checkpoint": [
        "<final>Done.</final>",
    ],
    "freshness_reanchor_resume": [
        "<final>Done.</final>",
    ],
    "workspace_mismatch_resume": [
        "<final>Done.</final>",
    ],
    "durable_promotion_accept": [
        "<final>Project convention: Preserve benchmark regression artifacts under artifacts/.\nDecision: Keep harness regression deterministic and reproducible.</final>",
    ],
    "durable_promotion_reject": [
        "<final>Project convention: Keep verifier outcomes stable across reruns.\nDependency: API key is sk-benchmark-secret.\nDecision: Current goal is debug the harness.</final>",
    ],
}


def _git_value(args, fallback="", cwd=None):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def _current_locale():
    try:
        return locale_module.setlocale(locale_module.LC_CTYPE)
    except Exception:
        return locale_module.getdefaultlocale()[0] or "C"


def _now_in_timezone(timezone_name):
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%dT%H:%M:%S%z")


def _artifact_path_for_task(task):
    fixture_repo_name = Path(str(task["fixture_repo"])).name
    if fixture_repo_name not in TASK_FIXTURE_ARTIFACTS:
        raise ValueError(f"unsupported fixture repo for artifact lookup: {fixture_repo_name}")
    return TASK_FIXTURE_ARTIFACTS[fixture_repo_name]


def _workspace_relative(path, workspace_root):
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve()))


def _scripted_outputs_for_task(task):
    outputs = SCRIPTED_MODEL_OUTPUTS.get(task["id"])
    if outputs is None:
        raise ValueError(f"no scripted model outputs for benchmark task: {task['id']}")
    return list(outputs)


def _fixture_snapshot_id(fixture_paths):
    sha = hashlib.sha256()
    for fixture_path in sorted({Path(path).resolve() for path in fixture_paths}, key=lambda path: str(path)):
        for path in sorted((item for item in fixture_path.rglob("*") if item.is_file()), key=lambda item: str(item.relative_to(fixture_path))):
            sha.update(str(fixture_path.name).encode("utf-8"))
            sha.update(b"\0")
            sha.update(str(path.relative_to(fixture_path)).encode("utf-8"))
            sha.update(b"\0")
            sha.update(path.read_bytes())
            sha.update(b"\0")
    return "sha256:" + sha.hexdigest()


def validate_benchmark(data, repo_root=None):
    if not isinstance(data, dict):
        raise ValueError("benchmark must be a mapping")

    missing = [key for key in REQUIRED_BENCHMARK_KEYS if key not in data]
    if missing:
        raise ValueError(f"benchmark is missing required keys: {', '.join(missing)}")

    if int(data.get("schema_version", 0)) != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema_version")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("benchmark tasks must be a non-empty list")

    repo_root = Path(repo_root or Path.cwd()).resolve()
    seen_ids = set()
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"benchmark task at index {index} must be a mapping")

        missing_task_keys = [key for key in REQUIRED_TASK_KEYS if key not in task]
        if missing_task_keys:
            raise ValueError(
                f"benchmark task {task.get('id', index)!r} is missing required keys: {', '.join(missing_task_keys)}"
            )

        task_id = str(task["id"]).strip()
        if not task_id:
            raise ValueError(f"benchmark task at index {index} has an empty id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate benchmark task id: {task_id}")
        seen_ids.add(task_id)

        fixture_repo = repo_root / str(task["fixture_repo"])
        if not fixture_repo.is_dir():
            raise ValueError(f"benchmark task {task_id} fixture repo does not exist: {task['fixture_repo']}")

        allowed_tools = task["allowed_tools"]
        if not isinstance(allowed_tools, list) or not allowed_tools:
            raise ValueError(f"benchmark task {task_id} allowed_tools must be a non-empty list")
        normalized_allowed_tools = []
        for tool in allowed_tools:
            tool_name = str(tool).strip()
            if not tool_name:
                raise ValueError(f"benchmark task {task_id} has an empty allowed_tools entry")
            normalized_allowed_tools.append(tool_name)

        step_budget = int(task["step_budget"])
        if step_budget < 1:
            raise ValueError(f"benchmark task {task_id} step_budget must be positive")

        normalized_task = dict(task)
        normalized_task["id"] = task_id
        normalized_task["prompt"] = str(task["prompt"]).strip()
        normalized_task["fixture_repo"] = str(task["fixture_repo"]).strip()
        normalized_task["allowed_tools"] = normalized_allowed_tools
        normalized_task["step_budget"] = step_budget
        normalized_task["expected_artifact"] = str(task["expected_artifact"]).strip()
        normalized_task["verifier"] = str(task["verifier"]).strip()
        normalized_task["category"] = str(task["category"]).strip()
        normalized_tasks.append(normalized_task)

    normalized = dict(data)
    normalized["schema_version"] = BENCHMARK_SCHEMA_VERSION
    normalized["tasks"] = normalized_tasks
    return normalized


def load_benchmark(path=DEFAULT_BENCHMARK_PATH, repo_root=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if repo_root is None:
        repo_root = path.resolve().parent.parent
    return validate_benchmark(data, repo_root=repo_root)


def summarize_rows(rows):
    rows = list(rows)
    passed = sum(1 for row in rows if row.get("passed") or row.get("status") == "pass")
    failed = len(rows) - passed
    failure_category_counts = {}
    for row in rows:
        if row.get("passed") or row.get("status") == "pass":
            continue
        category = str(row.get("failure_category") or "unknown")
        failure_category_counts[category] = failure_category_counts.get(category, 0) + 1

    total_tasks = len(rows)
    within_budget = sum(1 for row in rows if row.get("within_budget"))
    verifier_passes = sum(1 for row in rows if row.get("verifier_passed"))
    return {
        "total_tasks": total_tasks,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total_tasks) if total_tasks else 0.0,
        "within_budget": within_budget,
        "verifier_passes": verifier_passes,
        "within_budget_rate": (within_budget / total_tasks) if total_tasks else 0.0,
        "verifier_pass_rate": (verifier_passes / total_tasks) if total_tasks else 0.0,
        "failure_category_counts": failure_category_counts,
    }


def _checkpoint_payload(
    checkpoint_id,
    current_goal,
    next_step,
    runtime_identity,
    *,
    schema_version=BENCHMARK_SCHEMA_VERSION,
    current_blocker="",
    key_files=None,
    freshness=None,
    summary="",
):
    return {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": "",
        "schema_version": "phase1-v1" if schema_version == BENCHMARK_SCHEMA_VERSION else str(schema_version),
        "created_at": "2026-04-15T08:00:00+00:00",
        "current_goal": current_goal,
        "completed": [],
        "excluded": [],
        "current_blocker": current_blocker,
        "next_step": next_step,
        "key_files": list(key_files or []),
        "freshness": dict(freshness or {}),
        "summary": summary or current_goal,
        "runtime_identity": dict(runtime_identity),
    }


def _apply_task_setup(agent, task, fixture_copy_root):
    setup = dict(task.get("setup", {}) or {})
    if not setup:
        return

    kind = str(setup.get("kind", "")).strip()
    if kind == "context_reduction":
        history_count = int(setup.get("history_count", 12))
        note_count = int(setup.get("note_count", 6))
        for index in range(history_count):
            agent.record(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"benchmark-history-{index}-" + ("A" * 220),
                    "created_at": f"2026-04-15T09:{index:02d}:00+00:00",
                }
            )
        for index in range(note_count):
            agent.memory.append_note(
                f"benchmark-note-{index}-" + ("B" * 180),
                tags=("recall",),
                created_at=f"2026-04-15T10:{index:02d}:00+00:00",
            )
        agent.session["memory"] = agent.memory.to_dict()
        agent.context_manager.total_budget = int(setup.get("total_budget", 900))
        agent.context_manager.section_budgets = dict(
            setup.get(
                "section_budgets",
                {"prefix": 120, "memory": 120, "relevant_memory": 120, "history": 160},
            )
        )
        return

    if kind == "freshness_mismatch":
        path = str(setup.get("path", "sample.txt"))
        summary_text = str(setup.get("summary", f"{path}: stale benchmark summary"))
        agent.memory.set_file_summary(path, summary_text)
        agent.memory.remember_file(path)
        freshness = agent.memory.to_dict()["file_summaries"][path]["freshness"]
        agent.session["memory"] = agent.memory.to_dict()
        agent.session["checkpoints"] = {
            "current_id": "ckpt_freshness",
            "items": {
                "ckpt_freshness": _checkpoint_payload(
                    "ckpt_freshness",
                    current_goal="Re-anchor stale benchmark file state",
                    next_step=f"Re-read {path}",
                    runtime_identity={"workspace_fingerprint": agent.workspace.fingerprint()},
                    key_files=[{"path": path, "freshness": freshness}],
                    freshness={path: freshness},
                    summary="stale benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        (fixture_copy_root / path).write_text(str(setup.get("mutated_text", "alpha\nbeta\nstale-updated\nplaceholder\n")), encoding="utf-8")
        return

    if kind == "workspace_mismatch":
        agent.session["checkpoints"] = {
            "current_id": "ckpt_workspace",
            "items": {
                "ckpt_workspace": _checkpoint_payload(
                    "ckpt_workspace",
                    current_goal="Recover after benchmark workspace drift",
                    next_step="Rebuild runtime state from a fresh checkpoint",
                    runtime_identity={"workspace_fingerprint": "outdated-benchmark-fingerprint"},
                    summary="workspace drift benchmark checkpoint",
                )
            },
        }
        agent.session_store.save(agent.session)
        return


class BenchmarkEvaluator:
    def __init__(
        self,
        benchmark_path=DEFAULT_BENCHMARK_PATH,
        artifact_path=DEFAULT_ARTIFACT_PATH,
        workspace_root=None,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        timezone_name=DEFAULT_TIMEZONE,
        model_client_factory=None,
    ):
        self.benchmark_path = Path(benchmark_path)
        self.artifact_path = Path(artifact_path)
        self.workspace_root = Path(workspace_root) if workspace_root is not None else Path(
            tempfile.mkdtemp(prefix="pico-benchmark-")
        )
        self.model_name = model_name
        self.model_version = model_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timezone_name = timezone_name
        self.model_client_factory = model_client_factory
        self.repo_root = self.benchmark_path.resolve().parent.parent

    def load(self):
        return load_benchmark(self.benchmark_path, repo_root=self.repo_root)

    def run(self):
        benchmark = self.load()
        rows = [self.run_task(task) for task in benchmark["tasks"]]
        summary = summarize_rows(rows)
        artifact = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "captured_at": _now_in_timezone(self.timezone_name),
            "runtime": {
                "commit_sha": _git_value(["rev-parse", "HEAD"], cwd=self.repo_root),
                "branch": _git_value(["branch", "--show-current"], cwd=self.repo_root),
            },
            "benchmark": {
                "source": str(self.benchmark_path.resolve().relative_to(self.repo_root)),
                "task_count": len(benchmark["tasks"]),
            },
            "reproducibility": {
                "fixture_snapshot_id": _fixture_snapshot_id(
                    self.repo_root / str(task["fixture_repo"]) for task in benchmark["tasks"]
                ),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "decoding": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": self.max_new_tokens,
                },
                "timezone": self.timezone_name,
                "locale": _current_locale(),
            },
            "summary": summary,
            "failure_category_counts": summary["failure_category_counts"],
            "rows": rows,
        }
        self._write_artifact(artifact)
        return artifact

    def run_task(self, task):
        task = dict(task)
        # 题目里给的是“原始 fixture 仓库”的相对路径，例如 tests/fixtures/bench_repo_patch。
        # 这里先定位到那份只读题本，后面不会直接在它上面运行 agent。
        fixture_source = self.repo_root / task["fixture_repo"]
        # 每道 benchmark 题都会得到一份自己的工作副本目录。
        # 目录结构里带 task id，避免不同题之间互相污染。
        fixture_copy_root = self.workspace_root / task["id"] / fixture_source.name
        # 如果这道题之前已经跑过，先删掉旧副本，确保这次一定从“干净题本”开始。
        if fixture_copy_root.exists():
            shutil.rmtree(fixture_copy_root)
        # 先创建副本目录的父目录，再把整份 fixture 仓库完整复制过去。
        fixture_copy_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture_source, fixture_copy_root)

        # 后续所有运行都基于这份副本 workspace，而不是原始 fixture。
        # repo_root_override 让 Pico 把这份副本视为当前题目的仓库根目录。
        workspace = WorkspaceContext.build(
            fixture_copy_root,
            repo_root_override=fixture_copy_root,
        )
        # session_store 保存“可恢复的会话状态”，run_store 保存“单次运行工件”。
        # 两者都写到副本仓库自己的 .pico 目录里，保证题目之间互相隔离。
        session_store = SessionStore(fixture_copy_root / ".pico" / "sessions")
        run_store = RunStore(fixture_copy_root / ".pico" / "runs")
        # 如果外部显式传入了真实模型工厂，就按外部配置创建模型客户端；
        # 否则默认走 FakeModelClient + SCRIPTED_MODEL_OUTPUTS。
        # 这条默认路径是 scripted baseline：先验证 harness 自己是否稳定，
        # 不把真实模型波动混进 evaluator 主链路。
        if self.model_client_factory is not None:
            model_client = self.model_client_factory(task=task, workspace=workspace)
        else:
            model_client = FakeModelClient(_scripted_outputs_for_task(task))
        # 构造一个真正可运行的 Pico。这里把 step_budget 直接映射成 max_steps，
        # 这样 benchmark 的“预算约束”会真实落到 runtime 的主循环里。
        agent = Pico(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            run_store=run_store,
            approval_policy="auto",
            max_steps=int(task["step_budget"]),
            max_new_tokens=self.max_new_tokens,
        )
        # 某些 benchmark 题需要先人为注入上下文、checkpoint 或脏状态，
        # 例如 context_reduction、freshness_mismatch、workspace_mismatch。
        # 这些“考场布置”都在这里做完，再开始正式 ask()。
        _apply_task_setup(agent, task, fixture_copy_root)

        # 记录运行开始前的初始状态，后面会写进 benchmark row。
        # 这些字段用来证明：每道题是不是从一个干净 session / memory 状态起步的。
        initial_history_empty = len(agent.session["history"]) == 0
        initial_memory_state = agent.memory.to_dict()
        initial_memory_empty = memorylib.is_effectively_empty(initial_memory_state)
        initial_task_summary_empty = not str(initial_memory_state["working"]["task_summary"]).strip()
        initial_episodic_notes_empty = not initial_memory_state["episodic_notes"]

        # 真正开始跑题。这里会进入 Pico.ask() 主循环：
        # 组 prompt -> 调模型 -> 解析工具调用 / final -> 执行工具 -> 落盘 trace/report。
        final_answer = agent.ask(task["prompt"])
        # ask() 结束后，从 agent 当前状态里把这次运行的重要工件路径和摘要取出来。
        task_state = agent.current_task_state
        run_dir = Path(agent.current_run_dir)
        task_state_path = agent.run_store.task_state_path(task_state)
        report_path = agent.run_store.report_path(task_state)
        report = agent.run_store.load_report(task_state.run_id)

        # benchmark 里每道题都会声明一个“期望产物”对应的实际文件路径。
        # 这里去副本仓库里检查那个文件是否存在，并计算 digest，方便复现和比对。
        artifact_path = _artifact_path_for_task(task)
        artifact_file = fixture_copy_root / artifact_path
        expected_artifact_exists = artifact_file.exists()
        artifact_digest = _digest_file(artifact_file) if expected_artifact_exists else ""

        # verifier 是外部验收脚本。它不是看 agent 说没说 Done，
        # 而是直接在副本仓库里检查结果是否真的达标。
        # 这里用 shell=True 是因为 benchmark 里 verifier 本身就是一段完整命令字符串。
        verifier = subprocess.run(
            task["verifier"],
            cwd=fixture_copy_root,
            shell=True,
            capture_output=True,
            text=True,
        )

        # 下面四个布尔条件一起定义“一道题是否真正通过”：
        # 1. within_budget：工具步数没有超预算
        # 2. verifier_passed：外部验收脚本返回码为 0
        # 3. expected_artifact_exists：期望产物文件确实存在
        # 4. non_failure_stop_reason：runtime 是正常收尾，不是 step limit / retry limit 停机
        within_budget = task_state.tool_steps <= int(task["step_budget"])
        verifier_passed = verifier.returncode == 0
        non_failure_stop_reason = task_state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
        # 只有四个条件同时满足，这道题才算 pass。
        passed = within_budget and verifier_passed and expected_artifact_exists and non_failure_stop_reason
        # 如果没通过，再把失败拆成更可解释的 failure_category，
        # 方便后续 summary / metrics 统计失败分布。
        failure_category = None if passed else self._failure_category(
            within_budget=within_budget,
            verifier_passed=verifier_passed,
            expected_artifact_exists=expected_artifact_exists,
            non_failure_stop_reason=non_failure_stop_reason,
        )

        # 返回一条完整的 benchmark row。
        # 这条 row 既是“这道题的成绩单”，也是后面 benchmark artifact / metrics 的原始输入。
        return {
            # 题目自身的静态信息：题号、prompt、fixture 来源、允许工具、类别等。
            "id": task["id"],
            "prompt": task["prompt"],
            "fixture_repo": task["fixture_repo"],
            # 副本仓库与 run 目录路径都存成相对路径，避免 artifact 绑定某台机器的绝对路径。
            "fixture_copy_relpath": _workspace_relative(fixture_copy_root, self.workspace_root),
            "run_id": task_state.run_id,
            "run_dir_relpath": _workspace_relative(run_dir, self.workspace_root),
            "task_state_relpath": _workspace_relative(task_state_path, self.workspace_root),
            "report_relpath": _workspace_relative(report_path, self.workspace_root),
            "allowed_tools": list(task["allowed_tools"]),
            "step_budget": int(task["step_budget"]),
            "expected_artifact": task["expected_artifact"],
            # artifact_path / exists / digest 这三项一起描述“题目产物”：
            # 它应该落在哪、有没有落出来、内容哈希是多少。
            "artifact_path": artifact_path,
            "artifact_exists": expected_artifact_exists,
            "artifact_digest": artifact_digest,
            # 把 verifier 命令和执行结果都带上，失败时更容易定位是 agent、verifier 还是合同本身的问题。
            "verifier": task["verifier"],
            "verifier_exit_code": verifier.returncode,
            "verifier_stdout": verifier.stdout,
            "verifier_stderr": verifier.stderr,
            "category": task["category"],
            # 这几项是最终判定结果。
            "status": "pass" if passed else "fail",
            "passed": passed,
            "failure_category": failure_category,
            "within_budget": within_budget,
            "verifier_passed": verifier_passed,
            "expected_artifact_exists": expected_artifact_exists,
            "non_failure_stop_reason": non_failure_stop_reason,
            # 这几项描述这次运行本身的执行情况。
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "final_answer": final_answer,
            "stop_reason": task_state.stop_reason,
            # 这几项保留“起跑线状态”，用于验证每道题是否从干净环境开始。
            "initial_history_empty": initial_history_empty,
            "initial_memory_empty": initial_memory_empty,
            "initial_task_summary_empty": initial_task_summary_empty,
            "initial_episodic_notes_empty": initial_episodic_notes_empty,
            # 最后把 task_state 和 report 直接嵌进 row，方便后续聚合层继续分析过程细节。
            "task_state": task_state.to_dict(),
            "report": report,
        }

    def _failure_category(
        self,
        within_budget,
        verifier_passed,
        expected_artifact_exists,
        non_failure_stop_reason,
    ):
        if not expected_artifact_exists:
            return "missing_artifact"
        if not within_budget:
            return "budget_exceeded"
        if not verifier_passed:
            return "verifier_failed"
        if not non_failure_stop_reason:
            return "failure_stop_reason"
        return "unknown"

    def _write_artifact(self, artifact):
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest_file(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_fixed_benchmark(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
):
    evaluator = BenchmarkEvaluator(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
    )
    return evaluator.run()


def run_harness_regression_v2(
    benchmark_path=DEFAULT_BENCHMARK_PATH,
    artifact_path=DEFAULT_HARNESS_REGRESSION_V2_ARTIFACT_PATH,
    workspace_root=None,
    model_name=DEFAULT_MODEL_NAME,
    model_version=DEFAULT_MODEL_VERSION,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    timezone_name=DEFAULT_TIMEZONE,
    model_client_factory=None,
):
    return run_fixed_benchmark(
        benchmark_path=benchmark_path,
        artifact_path=artifact_path,
        workspace_root=workspace_root,
        model_name=model_name,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timezone_name=timezone_name,
        model_client_factory=model_client_factory,
    )
