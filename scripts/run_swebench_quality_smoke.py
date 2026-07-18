#!/usr/bin/env python3
"""Run Mio's sealed, non-evidence paired SWE-bench smoke locally.

This command is intentionally narrower than the generation library.  It loads
one private v1 smoke schedule, resolves every repository to an already-present
local bare mirror, forces the frozen target-only Qwen 3.6 27B controls onto the
exact :class:`~mio.config.TierConfig` loaded by ``ModelManager``, and seals the
content-free generation receipt.  It never downloads a model or dataset and it
never invokes the SWE-bench evaluator.

Only hashes, counts, and protocol status are written to stdout.  Model, issue,
tool, and patch text remain inside the private generation layout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import bench_swebench_quality as protocol  # noqa: E402
from scripts import run_swebench_quality_generation as generation  # noqa: E402

SMOKE_CLI_SCHEMA = f"{generation.GENERATION_SCHEMA}.smoke-cli.v1"
_LAYOUT_MODES = frozenset({"new", "resume"})
_OFFLINE_ENVIRONMENT = MappingProxyType(
    {
        "DO_NOT_TRACK": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "MIO_DDTREE_BUDGET": "0",
        "MIO_DRAFTER_STRICT": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
)


@dataclass(frozen=True)
class SmokeOptions:
    """Validated-by-construction command inputs, before filesystem preflight."""

    schedule_path: Path
    layout_root: Path
    layout_mode: str
    model_root: Path
    config_path: Path
    tier_name: str
    repo_source_arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.layout_mode not in _LAYOUT_MODES:
            raise protocol.ProtocolError("smoke layout mode must be explicitly new or resume")
        if not self.tier_name or self.tier_name.strip() != self.tier_name:
            raise protocol.ProtocolError("smoke tier name must be non-empty and canonical")
        if not self.repo_source_arguments:
            raise protocol.ProtocolError("at least one local repository source is required")


@dataclass(frozen=True)
class SmokeDependencies:
    """Dependency seam used by unit tests; the CLI always selects production."""

    load_schedule: Callable[[Path], tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]]
    load_config: Callable[[Path], Any]
    binding_factory: Callable[..., Any]
    manager_factory: Callable[[Any], Any]
    executor_factory: Callable[..., Any]
    workspace_factory: Callable[[Callable[[protocol.PublicInstance], Path]], Any]
    run_pairs: Callable[..., str]
    build_tool_surface: Callable[[], tuple[Mapping[str, Any], tuple[dict[str, Any], ...], str]]
    seal_receipt: Callable[..., str]
    verify_receipt: Callable[..., str]


@dataclass(frozen=True)
class SmokeResult:
    """Content-free terminal result safe to print or persist in CI logs."""

    layout_mode: str
    schedule_sha256: str
    factor_sha256: str
    receipt_sha256: str
    pair_count: int
    arm_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SMOKE_CLI_SCHEMA,
            "status": "sealed_non_evidence_smoke",
            "evidence_class": "non_evidence_smoke",
            "layout_mode": self.layout_mode,
            "schedule_sha256": self.schedule_sha256,
            "factor_sha256": self.factor_sha256,
            "receipt_sha256": self.receipt_sha256,
            "pair_count": self.pair_count,
            "arm_count": self.arm_count,
            "contains_issue_model_or_patch_text": False,
            "evaluator_invoked": False,
            "implicit_network_or_download_invoked": False,
            "confirmatory_evidence_admissible": False,
        }


def production_dependencies() -> SmokeDependencies:
    """Resolve production classes lazily so importing this CLI never loads MLX."""

    from mio.config import load_config
    from mio.model_manager import ModelManager

    return SmokeDependencies(
        load_schedule=protocol.load_private_schedule,
        load_config=load_config,
        binding_factory=generation.GenerationBinding.automatic_local,
        manager_factory=ModelManager,
        executor_factory=generation.NativeMioArmExecutor,
        workspace_factory=generation.ExternalGitWorkspaceFactory,
        run_pairs=generation.run_generation_pairs,
        build_tool_surface=generation.build_identical_tool_surface,
        seal_receipt=generation.seal_generation_receipt,
        verify_receipt=generation.verify_generation_receipt,
    )


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise protocol.ProtocolError(f"{label} is not a lowercase SHA-256")
    return value


def _require_exact_spelling(path: Path, label: str) -> Path:
    """Reject symlink, ``..``, relative, and case-folded filesystem aliases."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise protocol.ProtocolError(f"{label} must be an absolute canonical path")
    protocol._reject_symlink_path_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise protocol.ProtocolError(f"{label} does not exist") from exc
    if candidate != resolved:
        raise protocol.ProtocolError(f"{label} must not use a filesystem alias")

    current = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        try:
            names = {entry.name for entry in os.scandir(current)}
        except OSError as exc:
            raise protocol.ProtocolError(f"cannot inspect canonical {label}") from exc
        if component not in names:
            raise protocol.ProtocolError(f"{label} uses a filesystem spelling alias")
        current /= component
    return resolved


def _canonical_regular_file(path: Path, label: str) -> Path:
    resolved = _require_exact_spelling(path, label)
    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise protocol.ProtocolError(f"cannot inspect {label}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise protocol.ProtocolError(f"{label} must be a regular single-link file")
    return resolved


def _new_layout_root(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise protocol.ProtocolError("new layout root must be an absolute canonical path")
    protocol._reject_symlink_path_components(candidate)
    if os.path.lexists(candidate):
        raise protocol.ProtocolError("new layout root already exists; use explicit resume")
    parent = _require_exact_spelling(candidate.parent, "new layout parent")
    canonical = parent / candidate.name
    if candidate != canonical or not candidate.name or candidate.name in {".", ".."}:
        raise protocol.ProtocolError("new layout root must not use a filesystem alias")
    return canonical


def _stable_bytes(path: Path, label: str) -> bytes:
    payload = protocol._read_immutable_file(path)
    if payload is None:
        raise protocol.ProtocolError(f"{label} disappeared during preflight")
    return payload


def load_stable_schedule(
    path: Path,
    loader: Callable[[Path], tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]],
) -> tuple[dict[str, Any], tuple[protocol.ScheduleEntry, ...]]:
    """Load one canonical private schedule and detect concurrent replacement."""

    schedule_path = _canonical_regular_file(path, "private schedule")
    before = _stable_bytes(schedule_path, "private schedule")
    document, schedule = loader(schedule_path)
    after = _stable_bytes(schedule_path, "private schedule")
    if before != after:
        raise protocol.ProtocolError("private schedule changed while it was loaded")
    if protocol.canonical_json_bytes(document) != before:
        raise protocol.ProtocolError("private schedule must use canonical JSON encoding")
    if document.get("evidence_class") != "non_evidence_smoke":
        raise protocol.ProtocolError("this command accepts non-evidence smoke schedules only")
    if not schedule or len(schedule) % 2:
        raise protocol.ProtocolError("non-evidence smoke schedule must contain complete pairs")
    return document, schedule


def _load_stable_config(path: Path, loader: Callable[[Path], Any]) -> Any:
    config_path = _canonical_regular_file(path, "Mio config")
    if config_path.stat().st_mode & 0o077:
        raise protocol.ProtocolError("private Mio config must use 0600 permissions")
    before = _stable_bytes(config_path, "Mio config")
    try:
        raw = json.loads(before)
    except (UnicodeDecodeError, ValueError) as exc:
        raise protocol.ProtocolError("Mio config is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise protocol.ProtocolError("Mio config must be a JSON object")
    config = loader(config_path)
    if _stable_bytes(config_path, "Mio config") != before:
        raise protocol.ProtocolError("Mio config changed while it was loaded")
    return config


def _git_mirror_command(source: Path, argv: Sequence[str], *, allow_absent: bool = False) -> bytes:
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "TMPDIR": "/tmp",
    }
    result = subprocess.run(
        [
            protocol._trusted_git(),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "submodule.recurse=false",
            f"--git-dir={source}",
            *argv,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env=environment,
    )
    allowed = {0, 1} if allow_absent else {0}
    if result.returncode not in allowed:
        raise protocol.ProtocolError(f"local source mirror failed Git preflight ({argv[0]})")
    return result.stdout


def _validate_bare_source(source: Path, commits: Sequence[str]) -> None:
    try:
        is_bare = _git_mirror_command(source, ["rev-parse", "--is-bare-repository"]).strip()
    except protocol.ProtocolError as exc:
        raise protocol.ProtocolError("repository source must be a local bare or mirror clone") from exc
    if is_bare != b"true":
        raise protocol.ProtocolError("repository source must be a local bare or mirror clone")
    if _git_mirror_command(
        source,
        ["config", "--get", "extensions.partialClone"],
        allow_absent=True,
    ).strip():
        raise protocol.ProtocolError("partial/promisor source mirrors are not allowed")
    if _git_mirror_command(
        source,
        ["config", "--get-regexp", r"^remote\..*\.promisor$"],
        allow_absent=True,
    ).strip():
        raise protocol.ProtocolError("partial/promisor source mirrors are not allowed")
    if os.path.lexists(source / "objects" / "info" / "alternates"):
        raise protocol.ProtocolError("source mirror must not depend on an external object alternate")
    for commit in sorted(set(commits)):
        result = _git_mirror_command(source, ["cat-file", "-t", f"{commit}^{{commit}}"])
        if result.strip() != b"commit":
            raise protocol.ProtocolError("source mirror does not contain a scheduled base commit")


def resolve_repo_sources(
    arguments: Sequence[str],
    schedule_document: Mapping[str, Any],
) -> Mapping[str, Path]:
    """Build an exact one-to-one schedule-repository to local-mirror map."""

    raw_instances = schedule_document.get("public_instances")
    if not isinstance(raw_instances, list):
        raise protocol.ProtocolError("private schedule lacks public instances")
    instances = tuple(protocol.PublicInstance.from_mapping(row) for row in raw_instances)
    commits_by_repo: dict[str, set[str]] = {}
    for instance in instances:
        commits_by_repo.setdefault(instance.repo, set()).add(instance.base_commit)

    parsed: dict[str, Path] = {}
    used_sources: set[Path] = set()
    for argument in arguments:
        repo, separator, raw_path = argument.partition("=")
        if not separator or not repo or not raw_path or repo.strip() != repo:
            raise protocol.ProtocolError("repository sources must use exact REPO=/absolute/bare-mirror syntax")
        if repo in parsed:
            raise protocol.ProtocolError("repository source mapping contains a duplicate repository")
        source = _require_exact_spelling(Path(raw_path), "repository source mirror")
        source = protocol.require_private_path(source, must_exist=True)
        if not source.is_dir():
            raise protocol.ProtocolError("repository source mirror must be a directory")
        if source in used_sources:
            raise protocol.ProtocolError("repository source mirrors must be one-to-one with repositories")
        parsed[repo] = source
        used_sources.add(source)

    expected = set(commits_by_repo)
    if set(parsed) != expected:
        raise protocol.ProtocolError("repository source mapping is incomplete or contains an extra repository")
    for repo in sorted(expected):
        _validate_bare_source(parsed[repo], sorted(commits_by_repo[repo]))
    return MappingProxyType({repo: parsed[repo] for repo in sorted(parsed)})


def force_target_ar_controls(config: Any, tier_name: str, model_root: Path) -> Any:
    """Mutate the selected config tier into the preregistered 27B control."""

    tiers = getattr(config, "tiers", None)
    if not isinstance(tiers, dict) or tier_name not in tiers:
        raise protocol.ProtocolError("selected Mio tier is absent from the loaded configuration")
    tier = tiers[tier_name]
    frozen = {
        "target_model": str(model_root),
        "draft_model": "disabled-for-target-ar-smoke",
        "drafter_backend": "target_ar",
        "draft_fallback_model": None,
        "drafter_strict": True,
        "context_window": generation.TARGET_CONTEXT_TOKENS,
        "max_output_tokens": generation.TARGET_MAX_OUTPUT_TOKENS_PER_ROUND,
        "tq_bits": generation.TARGET_TQ_BITS,
        "pq_bits": generation.TARGET_PQ_BITS,
        "bmp_paths": generation.TARGET_BMP_PATHS,
        "ddtree_budget": generation.TARGET_DDTREE_BUDGET,
        "temperature": generation.TARGET_TEMPERATURE,
        "top_p": generation.TARGET_TOP_P,
        "top_k": generation.TARGET_TOP_K,
    }
    for name, value in frozen.items():
        if not hasattr(tier, name):
            raise protocol.ProtocolError(f"selected Mio tier lacks frozen field {name}")
        setattr(tier, name, value)
    config.active_tiers = [tier_name]
    config.tandem = False
    config.coding_effort = "medium"
    if tiers[tier_name] is not tier:
        raise protocol.ProtocolError("frozen tier was replaced while controls were applied")
    generation.validate_target_only_tier(tier)
    return tier


def _validate_loaded_engine(manager: Any, engine: Any, tier: Any, tier_name: str) -> None:
    if getattr(engine, "tier_config", None) is not tier:
        raise protocol.ProtocolError("loaded engine did not retain the frozen TierConfig object")
    generation.validate_target_only_tier(getattr(engine, "tier_config", None))
    if getattr(engine, "is_loaded", False) is not True:
        raise protocol.ProtocolError("target-only engine did not finish loading")
    if getattr(engine, "_target_model", None) is None or getattr(engine, "_tokenizer", None) is None:
        raise protocol.ProtocolError("target-only engine lacks target model or tokenizer")
    if getattr(engine, "_draft_model", None) is not None or getattr(engine, "_dspark_runtime", None) is not None:
        raise protocol.ProtocolError("target-only engine unexpectedly loaded a drafter")
    if (
        getattr(engine, "_drafter_requested", None) != "target_ar"
        or getattr(engine, "_drafter_selected", None) != "baseline"
        or getattr(engine, "_drafter_ref", None) is not None
    ):
        raise protocol.ProtocolError("loaded engine did not attest target_ar/baseline/no-drafter state")
    loaded_tiers = getattr(manager, "loaded_tiers", None)
    if not callable(loaded_tiers) or loaded_tiers() != [tier_name]:
        raise protocol.ProtocolError("model manager loaded tiers differ from the isolated smoke tier")
    if manager.get_engine(tier_name) is not engine:
        raise protocol.ProtocolError("model manager substituted the loaded target engine")


def _validate_last_raw_metrics(engine: Any) -> None:
    """Fail closed when the real smoke's final model round lacks raw timing.

    The generation runner currently returns aggregate arm outcomes rather than
    every ``AgentRoundTrace``.  This check therefore attests only the last
    exposed engine result and does not upgrade the smoke to evidence.  In
    particular, delivered-token budget accounting remains
    ``completion_tokens`` while decode work is ``physical_decode_tokens``.
    """

    metrics = getattr(engine, "last_metrics", None)
    if metrics is None:
        raise protocol.ProtocolError("smoke engine did not expose final generation metrics")
    if (
        getattr(metrics, "generation_backend", None) != "baseline"
        or getattr(metrics, "drafter_requested", None) != "target_ar"
        or getattr(metrics, "drafter_selected", None) != "baseline"
        or getattr(metrics, "drafter_ref", None) is not None
        or getattr(metrics, "fallback_ar", None) is not False
    ):
        raise protocol.ProtocolError("final smoke metrics differ from target_ar/baseline/no-drafter")
    if getattr(metrics, "timing_source", None) != "runtime_raw_ns":
        raise protocol.ProtocolError("final smoke metrics do not expose runtime_raw_ns timing")

    names = (
        "completion_tokens",
        "physical_decode_tokens",
        "logical_prompt_tokens",
        "physical_prefill_tokens",
        "warm_offset",
        "prefill_ns",
        "decode_ns",
        "model_total_ns",
    )
    values = {name: getattr(metrics, name, None) for name in names}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise protocol.ProtocolError("final smoke raw timing counters are incomplete")
    if values["physical_decode_tokens"] < values["completion_tokens"]:
        raise protocol.ProtocolError("physical decode work is below delivered completion tokens")
    if values["physical_prefill_tokens"] != values["logical_prompt_tokens"] - values["warm_offset"]:
        raise protocol.ProtocolError("final smoke physical prefill accounting is inconsistent")
    if values["model_total_ns"] != values["prefill_ns"] + values["decode_ns"]:
        raise protocol.ProtocolError("final smoke raw phase timing is inconsistent")


@contextlib.contextmanager
def _offline_environment():
    previous = {name: os.environ.get(name) for name in _OFFLINE_ENVIRONMENT}
    os.environ.update(_OFFLINE_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _discard_runtime_output():
    """Keep model/tool diagnostics out of the content-free CLI channel."""

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _prepare_layout_root(options: SmokeOptions) -> Path:
    if options.layout_mode == "new":
        return _new_layout_root(options.layout_root)
    root = _require_exact_spelling(options.layout_root, "resume layout root")
    return generation.GenerationLayout.open(root).root


def run_smoke(
    options: SmokeOptions,
    *,
    dependencies: SmokeDependencies | None = None,
) -> SmokeResult:
    """Execute and seal one production smoke, or a dependency-injected test run."""

    deps = dependencies or production_dependencies()
    schedule_document, schedule = load_stable_schedule(options.schedule_path, deps.load_schedule)
    sources = resolve_repo_sources(options.repo_source_arguments, schedule_document)
    layout_root = _prepare_layout_root(options)
    model_root = generation._canonical_local_directory(options.model_root, "local Qwen 3.6 27B model")
    config = _load_stable_config(options.config_path, deps.load_config)
    tier = force_target_ar_controls(config, options.tier_name, model_root)

    def source_for(instance: protocol.PublicInstance) -> Path:
        try:
            return sources[instance.repo]
        except KeyError as exc:
            raise protocol.ProtocolError("scheduled repository lacks a frozen local source") from exc

    with _offline_environment(), _discard_runtime_output():
        binding = deps.binding_factory(repository_root=ROOT, model_root=model_root)
        if getattr(binding, "model_identity", None) != protocol.EXPECTED_MODEL_IDENTITY:
            raise protocol.ProtocolError("automatic binding differs from the frozen 27B identity")
        layout = (
            generation.GenerationLayout.create(layout_root)
            if options.layout_mode == "new"
            else generation.GenerationLayout.open(layout_root)
        )
        manager = deps.manager_factory(config)
        try:
            manager.load_tier(options.tier_name)
            engine = manager.get_engine(options.tier_name)
            _validate_loaded_engine(manager, engine, tier, options.tier_name)
            executor = deps.executor_factory(
                engine=engine,
                manager=manager,
                config=config,
                tier=options.tier_name,
                require_raw_target_telemetry=True,
            )
            workspace_factory = deps.workspace_factory(source_for)
            _registry, _specs, surface_before = deps.build_tool_surface()
            _require_sha256(surface_before, "pre-run tool surface digest")
            pending_before = generation.pending_pairs(schedule, layout)
            factor_sha256 = deps.run_pairs(
                schedule_document=schedule_document,
                schedule=schedule,
                layout=layout,
                workspace_factory=workspace_factory,
                executor=executor,
                binding=binding,
                tier_config=tier,
            )
            _require_sha256(factor_sha256, "generation factor digest")
            if factor_sha256 != generation.factor_digest(surface_before):
                raise protocol.ProtocolError("generation factor differs from the frozen tool surface")
            if pending_before:
                _validate_last_raw_metrics(engine)
            _registry, _specs, surface_after = deps.build_tool_surface()
            if surface_after != surface_before:
                raise protocol.ProtocolError("native tool surface drifted during generation")
            binding.validate_for_run(
                evidence_run=False,
                executor=executor,
                tier_config=tier,
                require_executor_binding=True,
            )
            receipt_sha256 = deps.seal_receipt(
                schedule=schedule,
                layout=layout,
                binding=binding,
                tool_surface_sha256=surface_after,
                observed_model_identity_before=binding.model_identity,
                observed_model_identity_after=binding.model_identity,
            )
            _require_sha256(receipt_sha256, "sealed generation receipt digest")
            verified_sha256 = deps.verify_receipt(
                receipt_path=layout.receipt,
                schedule=schedule,
                layout=layout,
                binding=binding,
                tool_surface_sha256=surface_after,
            )
            if verified_sha256 != receipt_sha256:
                raise protocol.ProtocolError("sealed generation receipt did not verify byte-for-byte")
        finally:
            manager.unload_all()

    return SmokeResult(
        layout_mode=options.layout_mode,
        schedule_sha256=protocol.schedule_digest(schedule),
        factor_sha256=factor_sha256,
        receipt_sha256=receipt_sha256,
        pair_count=len(schedule) // 2,
        arm_count=len(schedule),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sealed local non-evidence paired SWE smoke on Qwen 3.6 27B.",
    )
    parser.add_argument("--schedule", type=Path, required=True, help="Canonical private v1 schedule (0600).")
    parser.add_argument("--layout", type=Path, required=True, help="Canonical private generation layout root.")
    layout = parser.add_mutually_exclusive_group(required=True)
    layout.add_argument("--new-layout", "--new", dest="layout_mode", action="store_const", const="new")
    layout.add_argument("--resume", dest="layout_mode", action="store_const", const="resume")
    parser.add_argument("--model", type=Path, required=True, help="Exact canonical local Qwen 3.6 27B MLX path.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Exact canonical existing private Mio config JSON (0600).",
    )
    parser.add_argument("--tier", required=True, help="Existing config tier to isolate and force to target_ar.")
    parser.add_argument(
        "--repo-source",
        action="append",
        required=True,
        metavar="REPO=/ABSOLUTE/BARE-MIRROR",
        help="Repeat once per distinct schedule repository; no network resolution is permitted.",
    )
    return parser


def _options_from_namespace(args: argparse.Namespace) -> SmokeOptions:
    return SmokeOptions(
        schedule_path=args.schedule,
        layout_root=args.layout,
        layout_mode=args.layout_mode,
        model_root=args.model,
        config_path=args.config,
        tier_name=args.tier,
        repo_source_arguments=tuple(args.repo_source),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(_options_from_namespace(args))
    except protocol.ProtocolError:
        print(
            protocol.canonical_json_bytes(
                {
                    "schema": SMOKE_CLI_SCHEMA,
                    "status": "rejected",
                    "error_code": "protocol_preflight_or_seal_failure",
                    "contains_issue_model_or_patch_text": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            protocol.canonical_json_bytes(
                {
                    "schema": SMOKE_CLI_SCHEMA,
                    "status": "failed",
                    "error_code": "runtime_failure",
                    "contains_issue_model_or_patch_text": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(protocol.canonical_json_bytes(result.as_dict()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
