# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive understanding
of the codebase by invoking the `/explore-codebase` skill.

This builds an accurate model of the project architecture before changes are made, keeping new code consistent with the
patterns that the library, its CLI, and its MCP server already follow.

## Style guide compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load
Ataraxis framework conventions. This applies to ALL file types:

| Task                                    | Skill to invoke    |
|-----------------------------------------|--------------------|
| Writing or modifying Python code        | `/python-style`    |
| Writing or modifying README files       | `/readme-style`    |
| Writing or modifying pyproject.toml     | `/pyproject-style` |
| Writing or modifying tox.ini files      | `/tox-config`      |
| Writing or modifying Sphinx docs files  | `/api-docs`        |
| Writing or modifying skill files        | `/skill-design`    |
| Creating or verifying project structure | `/project-layout`  |
| Committing local changes                | `/commit`          |

This is non-negotiable. Each skill carries a verification checklist that you MUST complete before submitting any work.

## Cross-referenced library verification

Ataraxis framework projects often depend on other `ataraxis-*` or `sollertia-*` libraries. These libraries may be
stored locally in the same parent directory as this project, reachable as `../` from the repository root.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

Skills are distributed through the ataraxis marketplace and are loaded into Claude Code via the plugin system. They are
**not** stored in this repository.

### Video plugin skills (ataraxis/plugins/video/)

| Skill                          | Description                                                          |
|--------------------------------|----------------------------------------------------------------------|
| `/camera-setup`                | MCP-based camera discovery, testing, encoding guidance, and GenICam  |
| `/camera-interface`            | VideoSystem API usage, constructor parameters, and encoding guidance |
| `/video-mcp-environment-setup` | MCP server connectivity diagnostics and environment verification     |
| `/post-recording`              | Post-recording verification: log assembly, video validation, handoff |
| `/pipeline`                    | End-to-end pipeline orchestration and multi-camera planning          |
| `/log-input-format`            | Reference for NPZ archive format, source IDs, and DataLogger output  |
| `/log-processing`              | Orchestrate log archive processing workflow via MCP tools            |
| `/log-processing-results`      | Reference for output data formats and frame statistics analysis      |

### Automation plugin skills relevant to this Python-only project (ataraxis/plugins/automation/)

| Skill                   | Description                                                                    |
|-------------------------|--------------------------------------------------------------------------------|
| `/explore-codebase`     | Perform in-depth codebase exploration at session start                         |
| `/explore-dependencies` | Explore installed ataraxis dependency APIs for reuse opportunities             |
| `/audit-correctness`    | Audit source code for bugs, edge cases, races, and leaks                       |
| `/audit-facts`          | Audit documentation for factual accuracy against source code                   |
| `/audit-performance`    | Audit source code for cost, speed, memory use, and dtype predictability        |
| `/audit-project`        | Orchestrate the four audits and merge their findings into one report           |
| `/audit-style`          | Audit files for style and convention compliance                                |
| `/python-style`         | Apply Ataraxis framework Python coding conventions (REQUIRED for code changes) |
| `/readme-style`         | Apply Ataraxis framework README conventions                                    |
| `/pyproject-style`      | Apply Ataraxis framework pyproject.toml conventions                            |
| `/tox-config`           | Apply Ataraxis framework tox.ini conventions                                   |
| `/api-docs`             | Apply Ataraxis framework Sphinx API documentation conventions                  |
| `/skill-design`         | Generate and verify Claude Code skill files                                    |
| `/project-layout`       | Apply Ataraxis framework project directory structure conventions               |
| `/commit`               | Draft Ataraxis framework style-compliant git commit messages                   |
| `/pr`                   | Draft Ataraxis framework style-compliant pull request summaries                |
| `/release`              | Draft Ataraxis framework style-compliant release notes                         |

## MCP server

This library provides an MCP server (`axvs mcp`) that exposes camera discovery, configuration, video recording,
manifest management, and log data processing tools. When working with this project or its dependencies, prefer using
available MCP tools over direct code execution when appropriate.

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools
   they provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like camera discovery, CTI file management, GenICam
   configuration, and log processing batch workflows, use MCP tools rather than writing and executing Python code
   directly. MCP tools provide:
   - Consistent, tested interfaces
   - Proper resource management and cleanup
   - Formatted output designed for user display

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when:
   - No MCP tool exists for the required functionality
   - The task requires custom logic not covered by available tools
   - Writing or modifying library source code

## Distribution model

This project follows a dual distribution model. The library source code, tests, CLI, and MCP server implementation live
in this repository (`ataraxis-video-system`) and are distributed via PyPI. Claude Code skills and MCP server
registration are distributed separately through the [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as
plugins:

- **video** plugin (`ataraxis/plugins/video/`): Registers the `axvs mcp` server with compatible MCP clients and
  provides video-specific skills for camera setup, pipeline orchestration, log processing, and post-recording
  verification.
- **automation** plugin (`ataraxis/plugins/automation/`): Provides shared development skills that enforce
  Ataraxis framework coding conventions (Python style, README style, commit messages, pyproject.toml, tox
  configuration) and general-purpose codebase exploration tools.

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository, not in this repository.
When modifying the MCP server implementation or library code, edit the source files in this repository.

## Project context

This is **ataraxis-video-system**, a Python library that interfaces with a wide range of cameras to flexibly record
visual stream data as video files. It supports OpenCV and GenICam (Harvesters) camera interfaces with FFMPEG-based
video encoding using CPU or GPU.

### Key areas

| Directory                             | Purpose                                                                           |
|---------------------------------------|-----------------------------------------------------------------------------------|
| `src/ataraxis_video_system/`          | Main library source code                                                          |
| `src/.../video/`                      | Camera acquisition, encoding, and timestamp extraction subpackage                 |
| `src/.../video/video_system.py`       | Core VideoSystem class with multiprocessing architecture                          |
| `src/.../video/camera.py`             | Camera interfaces (OpenCV, Harvesters, Mock) and CTI management                   |
| `src/.../video/saver.py`              | VideoSaver with FFMPEG subprocess encoding                                        |
| `src/.../video/configuration.py`      | GenICam node inspection, read/write, dump/load via YAML                           |
| `src/.../video/manifest.py`           | Camera manifest data classes and writer for source-to-name mappings               |
| `src/.../video/timestamps.py`         | Frame acquisition timestamp extraction algorithm                                  |
| `src/.../orchestration/`              | Job identity, sizing, discovery, the single-job runner, and both execution paths  |
| `src/.../orchestration/jobs.py`       | Job identity, the output layout enumeration, and the job descriptor               |
| `src/.../orchestration/allocation.py` | Declared core allocation and archive-derived memory footprint model               |
| `src/.../orchestration/discovery.py`  | Manifest-derived job resolution, preparation, and the per-job sizing pass         |
| `src/.../orchestration/worker.py`     | Single-job runner and the picklable descriptor-addressed pool entry point         |
| `src/.../orchestration/execution.py`  | Shared-pool batch engine admitting jobs against core and memory budgets           |
| `src/.../orchestration/pipeline.py`   | Sequential pipeline entry point for one recording                                 |
| `src/.../interfaces/`                 | CLI and MCP server subpackage                                                     |
| `src/.../interfaces/cli.py`           | Click-based `axvs` CLI with subcommand groups                                     |
| `src/.../interfaces/mcp_server.py`    | MCP server entry point that wires up tools and runs MCPServer                     |
| `src/.../interfaces/mcp_instance.py`  | Shared MCPServer instance and cross-tool helper functions                         |
| `src/.../interfaces/*_tools.py`       | 27 MCP tools (camera, session, configuration, discovery, processing)              |
| `tests/`                              | Test suite mirroring the video and orchestration subpackages, plus shared helpers |
| `docs/`                               | Sphinx API documentation source                                                   |

### Architecture

- **VideoSystem**: Producer-consumer multiprocessing pattern. Constructor requires `system_id`, `data_logger`,
  `name`, and `output_directory` as positional arguments. `__init__` writes a camera manifest entry associating
  the system_id with the human-readable name. A producer process acquires frames from the camera interface and
  pushes them to a queue. A consumer process pops frames and streams raw bytes to FFMPEG via stdin. A 4-element
  SharedMemoryArray controls termination and frame-saving toggles via IPC.
- **GenICam Platform Support**: `harvesters` and `genicam` carry a `sys_platform != 'darwin'` marker, because `genicam`
  publishes no macOS wheel for every supported Python version. macOS therefore does not support the GenICam camera
  interface at all. `camera.py` imports both inside a top-of-file `try/except ImportError` that falls back to `None`,
  keeping the module usable for the OpenCV and Mock interfaces there. `genicam_runtime_available()` reports the state,
  `_require_genicam_runtime()` aborts with `NotImplementedError` at every entry point that reaches GenICam hardware
  (`HarvestersCamera.connect()` and `add_cti_file()`), and `discover_camera_ids()` and `check_cti_file()` report the
  interface as absent instead of failing. A `[[tool.mypy.overrides]]` block marks both modules
  `ignore_missing_imports`, which is what keeps `tox -e lint` passing on a macOS host where neither is installed.
- **Camera Interfaces**: Three implementations behind a unified API: OpenCVCamera (cv2.VideoCapture),
  HarvestersCamera (GenICam/Harvesters with NodeMap access), and MockCamera (synthetic frames for testing).
  `discover_camera_ids()` returns CameraInformation objects from all available interfaces.
- **VideoSaver**: Manages an FFMPEG subprocess (Popen with stdin=PIPE). A daemon thread drains stderr continuously
  to prevent pipe buffer deadlocks. Supports CPU (libx264/libx265) and GPU (h264_nvenc/hevc_nvenc) encoders.
- **GenICam Configuration**: Iterative stack-based NodeMap traversal collects ReadWrite leaf nodes. Configurations
  serialize to YAML via GenicamConfiguration dataclass with camera identity metadata for validation on load.
- **Camera Manifest**: `CameraManifest` (YamlConfig subclass) associates source IDs with human-readable camera
  names in a `camera_manifest.yaml` file alongside DataLogger archives. `CameraSourceData` stores per-camera
  entries. `write_camera_manifest()` creates or updates the manifest. Used by log processing discovery to identify
  which archives were produced by ataraxis-video-system and to route processing by source ID.
- **Log Processing**: Extracts frame acquisition timestamps from DataLogger `.npz` archives, one job per archive.
  `resolve_jobs()` reads the recording's `camera_manifest.yaml` and reports an empty universe for a tree holding none,
  `prepare_jobs()` registers the resolved jobs without reading an archive, and `size_job()` applies the memory model in
  one pass. Uses `LogArchiveReader` for archive access and `ProcessingTracker` for job lifecycle management.
  `run_log_processing_pipeline()` runs one recording sequentially, or the single job a caller names by its canonical
  identifier, and fails when it resolves none. Outputs Feather files in a `camera_timestamps/` subdirectory.
- **MCP Server**: A shared `MCPServer` instance (`name="ataraxis-video-system"`) is defined in
  `interfaces/mcp_instance.py`, and `run_server()` enables JSON responses when it starts the streamable-http transport.
  Session global state (`_active_session`, `_active_logger`, `_session_info`), defined in `interfaces/session_tools.py`,
  enforces a single active VideoSystem session at a time. The 27 tools are split across the `interfaces/*_tools.py`
  modules. Tool categories: camera discovery and CTI management (3), system checks (1), video session lifecycle (5),
  GenICam configuration (4), camera manifest management (2), log archive and video validation (2), recording discovery
  via manifests (1), batch log processing execution (3), processing status and management (4), and post-processing
  analysis and cleanup (2). Session tools expose configurable encoding parameters (encoder, speed preset, pixel format,
  quantization). `stop_video_session` auto-assembles log archives and returns output paths. Batch log processing uses
  `JobExecutionState` (in `orchestration/execution.py`) with separate core and memory budgets. `size_job()` resolves
  each job's cores and memory from its own archive before dispatch, and `job_execution_manager()` admits what both
  budgets fit, running an oversized job alone so it never stalls the queue. The MCP server
  is registered with MCP clients via the **video** plugin in the ataraxis marketplace, not directly from this
  repository.
- **CLI**: Click command groups (`cti`, `check`, `configure`) with `run` for interactive sessions, `process` for
  log data processing, and `mcp` for starting the MCP server. CLI uses system_id 111, MCP uses 112.

### Key patterns

- **Multiprocessing Spawn**: `mp.set_start_method("spawn")` is set globally in `__init__.py` for cross-platform
  consistency. All spawned processes are daemon processes requiring explicit `stop()` calls.
- **FFMPEG Stderr Draining**: A dedicated thread reads FFMPEG's stderr continuously. Without this, the pipe buffer
  fills and blocks FFMPEG, deadlocking the encoding pipeline.
- **Camera Reconnection**: Cameras connect/disconnect during VideoSystem `__init__()` to validate parameters.
  The producer process reconnects the camera independently (fresh connection per process).
- **SharedMemoryArray IPC**: A 4-element uint8 array controls process lifecycle: index 0 = termination flag,
  index 1 = frame saving toggle, indices 2-3 = process initialization handshake signals.
- **CTI File Persistence**: The GenTL Producer path is stored in `platformdirs.user_data_dir` as `cti_path.txt`,
  persisting across runtimes.
- **Manifest-Based Log Discovery**: `camera_manifest.yaml` files tag DataLogger output directories with
  source-to-name mappings. Log processing discovery and batch preparation use manifests to identify which
  archives were produced by ataraxis-video-system and to route jobs by source ID.
- **Frame Display**: Runs on a separate thread with its own Queue (decoupled from saver). Automatically disabled
  on macOS due to main-thread GUI restrictions.

### Code standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow guidance

**Modifying VideoSystem:**

1. Review `src/ataraxis_video_system/video/video_system.py` for current implementation
2. Understand the producer-consumer multiprocessing architecture and SharedMemoryArray IPC
3. Static methods `_frame_production_loop()` and `_frame_saving_loop()` run in separate processes
4. Test with actual camera hardware or MockCamera interface

**Modifying camera interfaces:**

1. Review `src/ataraxis_video_system/video/camera.py` for all three implementations
2. OpenCVCamera and HarvestersCamera share a common interface pattern (connect, grab, disconnect)
3. CTI file management functions (`add_cti_file`, `check_cti_file`) use platformdirs for persistence
4. Camera discovery must handle both OpenCV and Harvesters gracefully when hardware is unavailable

**Modifying FFMPEG encoding:**

1. Review `src/ataraxis_video_system/video/saver.py` for the VideoSaver class
2. Encoder command construction happens in `__init__()` based on VideoEncoders enum selection
3. CPU presets (veryfast-veryslow) map to GPU p1-p7 equivalents
4. The stderr drain thread is critical and must not be removed

**Modifying GenICam configuration:**

1. Review `src/ataraxis_video_system/video/configuration.py` for node traversal and serialization
2. Node enumeration uses iterative stack-based traversal (not recursive)
3. GenicamConfiguration is a YamlConfig subclass supporting `to_yaml()` and `from_yaml()`
4. Strict identity checking compares camera model and serial number against YAML metadata

**Adding or modifying CLI commands:**

1. Review `src/ataraxis_video_system/interfaces/cli.py` for existing Click group structure
2. Follow existing patterns for option decorators and error handling
3. Use `console.echo()` for output and `console.error()` for error handling

**Modifying camera manifests:**

1. Review `src/ataraxis_video_system/video/manifest.py` for data classes and writer
2. `CameraManifest` extends `YamlConfig` from `ataraxis-data-structures` for YAML serialization
3. `CameraSourceData` is a frozen dataclass storing `id` (int) and `name` (str)
4. Manifests are consumed by `discover_camera_data_tool` in the MCP server for log processing discovery

**Modifying log processing:**

1. Review `orchestration/discovery.py` for resolution, preparation, and sizing, `orchestration/worker.py` for the
   single-job runner, and `video/timestamps.py` for the extraction algorithm
2. `extract_logged_camera_timestamps()` reads `.npz` archives via `LogArchiveReader` and returns `NDArray[np.uint64]`
3. `run_log_processing_pipeline()` runs one recording sequentially, or one job by identifier, and imports no engine
4. `ProcessingTracker` manages job lifecycle (SCHEDULED → RUNNING → SUCCEEDED/FAILED) via YAML state files
5. `_process_frame_message_batch()` runs in subprocess workers and is excluded from coverage (`# pragma: no cover`)
6. Log discovery uses manifest-based routing via `camera_manifest.yaml` files

**Adding or modifying MCP tools:**

1. Review the `interfaces/*_tools.py` modules for tool patterns (`interfaces/mcp_instance.py` holds the `mcp` instance)
2. Enforce single-session constraint via `_active_session` global state check
3. Log processing execution uses `JobExecutionState` (from `..orchestration`) with core and memory budgets
4. `size_job()` sizes each job from its own archive, and `job_execution_manager()` admits what both budgets fit
5. Most tools return `dict[str, Any]` structured data, and some return strings
6. MCP server registration happens in the ataraxis marketplace video plugin, not in this repository
