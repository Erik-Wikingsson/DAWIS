"""Path configuration.

Every machine-specific path is read from a variable. Resolution order, first
hit wins: the process environment, the repo `.env`, `data/paths.yaml`, and
then the default given by the caller (if any). See `.env.example`.
"""

import os
import re
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_LOADED = False
_DOTENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _find_dotenv() -> Optional[Path]:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidate = base / ".env"
        if candidate.is_file():
            return candidate

    repo_candidate = REPO_ROOT / ".env"
    if repo_candidate.is_file():
        return repo_candidate

    return None


def load_env_file() -> None:
    """Load `.env` into `os.environ` without overriding variables already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = _find_dotenv()
    if env_path is None:
        _ENV_LOADED = True
        return

    # Same rules as scripts/load_env.sh: the last assignment wins, quotes are
    # stripped, and an unquoted value ends at a trailing comment.
    values = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        match = _DOTENV_LINE.match(raw_line)
        if not match:
            continue
        key, value = match.groups()
        quoted = re.match(r"""^(["'])(.*?)\1""", value)
        values[key] = quoted.group(2) if quoted else value.split("#", 1)[0].rstrip()

    for key, value in values.items():
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    load_env_file()
    return os.environ.get(name, default)


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class MissingRootError(FileNotFoundError):
    """A path variable that is needed could not be resolved."""


def _how_to_set(var: str) -> str:
    return (f"Set it in any of:\n"
            f"  1. the environment:      export {var}=/path/to/...\n"
            f"  2. the repo .env file:   {var}=/path/to/...  (see .env.example)\n"
            f"  3. data/paths.yaml:      {var}: /path/to/...")


def _paths_yaml_values() -> dict:
    path = Path(__file__).resolve().parent / "paths.yaml"
    if not path.is_file():
        return {}
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lookup(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve one variable: environment, then `.env`, then `data/paths.yaml`."""
    load_env_file()
    if name in os.environ:
        return os.environ[name]
    from_yaml = _paths_yaml_values().get(name)
    if from_yaml is not None:
        return str(from_yaml)
    return default


def require(name: str, context: str = "") -> str:
    """Like `lookup`, but raise MissingRootError if the variable is unset."""
    value = lookup(name)
    if value in (None, ""):
        prefix = f"{context}: " if context else ""
        raise MissingRootError(f"{prefix}{name} is not set.\n{_how_to_set(name)}")
    return value


def expand(value, *, context: str = "config"):
    """Recursively expand `${VAR}` / `${VAR:-fallback}` in a nested structure."""
    if isinstance(value, str):

        def _sub(match: "re.Match") -> str:
            var, fallback = match.group(1), match.group(2)
            resolved = lookup(var, fallback)
            if resolved is None:
                raise MissingRootError(
                    f"{context}: path variable ${{{var}}} is not set.\n"
                    f"{_how_to_set(var)}\n"
                    f"  or pass --data_root /path/to/data")
            return resolved

        return _VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand(v, context=context) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, context=context) for v in value]
    return value


def resolve_root(spec: str, *, cli_override: Optional[str] = None,
                 context: str = "config") -> Path:
    """Resolve a config `root:` spec to a Path; `--data_root` beats everything."""
    if cli_override:
        return Path(cli_override).expanduser()
    return Path(expand(spec, context=context)).expanduser()


def models_path(subpath: str) -> str:
    """`subpath` under MODELS_ROOT, the directory holding the pretrained models."""
    return str(Path(require("MODELS_ROOT", context=f"checkpoint {subpath}"))
               .expanduser() / subpath)


def checkpoint(name: str, subpath: Optional[str] = None) -> Optional[str]:
    """Resolve a checkpoint from variable `name`, else `subpath` under MODELS_ROOT."""
    value = lookup(name)
    if value not in (None, ""):
        return value
    if subpath is None:
        return None
    return models_path(subpath)


#: Fallback results location, relative to the repo. Only consulted when reading.
LEGACY_RESULTS_DIR = "assimilation/results"

#: Group-writable modes, so results in a shared directory are readable by all.
RESULTS_DIR_MODE = 0o2775
RESULTS_FILE_MODE = 0o664


def results_dir(create: bool = False) -> Path:
    """The directory assimilation runs write their .nc results to (RESULTS_ROOT)."""
    path = Path(require("RESULTS_ROOT", context="results")).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
        share_path(path, RESULTS_DIR_MODE)
    return path


def share_path(path, mode: int) -> bool:
    """chmod `path` to `mode`; returns False instead of raising if not permitted."""
    try:
        os.chmod(path, mode)
        return True
    except OSError:
        return False


def result_file(name: str, create_dir: bool = False) -> Path:
    """The .nc file for experiment `name` in RESULTS_ROOT.

    Readers fall back to LEGACY_RESULTS_DIR when the file only exists there.
    """
    if not name.endswith(".nc"):
        name = f"{name}.nc"
    shared = results_dir(create=create_dir) / name
    if not create_dir and not shared.exists():
        for legacy in _legacy_result_dirs():
            candidate = legacy / name
            if candidate.exists():
                return candidate
    return shared


def _legacy_result_dirs():
    seen = []
    for base in (Path.cwd(), REPO_ROOT):
        candidate = base / LEGACY_RESULTS_DIR
        if candidate.is_dir() and candidate not in seen:
            seen.append(candidate)
    return seen


# Archived paper runs are named `<METHOD>_<EXPERIMENT>_run_<DATA_INDEX>_<date>_<uid>.nc`,
# so a run can be addressed by (method, experiment, data index) alone.


def paper_runs_dir() -> Path:
    """The directory holding the archived paper runs (PAPER_RUNS_ROOT)."""
    return Path(require("PAPER_RUNS_ROOT", context="paper runs")).expanduser()


def _paper_run_experiment_spellings(experiment: str):
    """Spellings of `experiment` in archive filenames (`saturating`, `Saturating`, `SEVIR`)."""
    seen = []
    for candidate in (experiment, experiment.capitalize(), experiment.lower(),
                      experiment.upper()):
        if candidate and candidate not in seen:
            seen.append(candidate)
    return seen


def find_paper_run(method: str, experiment: str, data_index) -> Path:
    """The archived run for `method` on `experiment`, trajectory `data_index`."""
    root = paper_runs_dir()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Paper-run directory not found: {root}\n"
            f"  Set PAPER_RUNS_ROOT in the environment, the repo .env, or "
            f"data/paths.yaml.")

    matches = []
    for spelling in _paper_run_experiment_spellings(experiment):
        matches = sorted(root.glob(f"{method}_{spelling}_run_{data_index}_*.nc"))
        if matches:
            break

    if not matches:
        available = sorted({
            p.name.split("_run_")[0][:-len(spelling) - 1]
            for spelling in _paper_run_experiment_spellings(experiment)
            for p in root.glob(f"*_{spelling}_run_{data_index}_*.nc")
        })
        detail = ("\n  Available for this experiment/data index: "
                  + ", ".join(available)) if available else (
                  "\n  No run at all matches this experiment/data index.")
        raise FileNotFoundError(
            f"No paper run named "
            f"'{method}_{experiment.capitalize()}_run_{data_index}_*.nc' in {root}."
            + detail)

    if len(matches) > 1:
        print(f"find_paper_run: {len(matches)} runs match "
              f"{method}/{experiment}/{data_index}; using {matches[-1].name}. "
              f"Others: {', '.join(p.name for p in matches[:-1])}")

    return matches[-1]
