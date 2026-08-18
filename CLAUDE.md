# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive understanding
of the codebase by invoking the `/explore-codebase` skill.

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

| Skill                          | Description                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------|
| `/camera-setup`                | Discovers and tests cameras over MCP, and guides encoding and GenICam configuration |
| `/camera-interface`            | Documents VideoSystem API usage, constructor parameters, and encoding guidance      |
| `/video-mcp-environment-setup` | Diagnoses MCP server connectivity and verifies the environment                      |
| `/cli-reference`               | Documents every `axvs` command, option, and failure mode                            |
| `/post-recording`              | Verifies recordings through log assembly, video validation, and handoff             |
| `/pipeline`                    | Orchestrates the end-to-end pipeline and multi-camera planning                      |
| `/log-input-format`            | Documents the NPZ archive format, source IDs, and DataLogger output                 |
| `/log-processing`              | Orchestrates the log archive processing workflow via MCP tools                      |
| `/log-processing-results`      | Documents output data formats and frame statistics analysis                         |

### Automation plugin skills relevant to this Python-only project (ataraxis/plugins/automation/)

| Skill                   | Description                                                                      |
|-------------------------|----------------------------------------------------------------------------------|
| `/explore-codebase`     | Performs in-depth codebase exploration at session start                          |
| `/explore-dependencies` | Explores installed ataraxis dependency APIs for reuse opportunities              |
| `/audit-correctness`    | Audits source code for bugs, edge cases, races, and leaks                        |
| `/audit-facts`          | Audits documentation for factual accuracy against source code                    |
| `/audit-performance`    | Audits source code for cost, speed, memory use, and dtype predictability         |
| `/audit-project`        | Orchestrates the four audits and merges their findings into one report           |
| `/audit-style`          | Audits files for style and convention compliance                                 |
| `/python-style`         | Applies Ataraxis framework Python coding conventions (REQUIRED for code changes) |
| `/readme-style`         | Applies Ataraxis framework README conventions                                    |
| `/pyproject-style`      | Applies Ataraxis framework pyproject.toml conventions                            |
| `/tox-config`           | Applies Ataraxis framework tox.ini conventions                                   |
| `/api-docs`             | Applies Ataraxis framework Sphinx API documentation conventions                  |
| `/skill-design`         | Generates and verifies Claude Code skill files                                   |
| `/project-layout`       | Applies Ataraxis framework project directory structure conventions               |
| `/commit`               | Drafts Ataraxis framework style-compliant git commit messages                    |
| `/pr`                   | Drafts Ataraxis framework style-compliant pull request summaries                 |
| `/release`              | Drafts Ataraxis framework style-compliant release notes                          |

## MCP server

This library provides an MCP server (`axvs mcp`) that exposes camera discovery, configuration, video recording,
manifest management, and log data processing tools. When working with this project or its dependencies, prefer using
available MCP tools over direct code execution when appropriate.

**Guidelines for MCP usage:**

1. **Prefer MCP for runtime operations**: For operations like camera discovery, CTI file management, GenICam
   configuration, and log processing batch workflows, use MCP tools rather than writing and executing Python code
   directly.

2. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

## Distribution model

The library source code, tests, CLI, and MCP server implementation live in this repository (`ataraxis-video-system`) and
are distributed via PyPI. Claude Code skills and MCP server registration are distributed separately through the
[ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as plugins:

- **video** plugin (`ataraxis/plugins/video/`): Registers the `axvs mcp` server with compatible MCP clients and
  provides video-specific skills for camera setup, pipeline orchestration, log processing, and post-recording
  verification.
- **automation** plugin (`ataraxis/plugins/automation/`): Provides shared development skills that enforce
  Ataraxis framework coding conventions (Python style, README style, commit messages, pyproject.toml, tox
  configuration) and general-purpose codebase exploration tools.

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository. When modifying the MCP server
implementation or library code, edit the source files in this repository.

## Project context

This is **ataraxis-video-system**, a Python library that interfaces with a wide range of cameras to flexibly record
visual stream data as video files. It supports OpenCV and GenICam (Harvesters) camera interfaces with FFMPEG-based
video encoding using CPU or GPU.

### Key areas

| Directory                    | Purpose                                                                                                             |
|------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `src/ataraxis_video_system/` | Main library source code                                                                                            |
| `src/.../video/`             | Camera interfaces, VideoSystem multiprocessing, FFMPEG encoding, GenICam configuration, manifests, and timestamps   |
| `src/.../orchestration/`     | Job identity and layout, core and memory allocation, discovery and sizing, the job runner, and both execution paths |
| `src/.../interfaces/`        | Click-based `axvs` CLI, the MCP server instance and its tools, and the shared response machinery                    |
| `tests/`                     | Test suite mirroring the video, orchestration, and interfaces subpackages                                           |
| `docs/`                      | Sphinx API documentation source                                                                                     |

### Architecture

- **VideoSystem**: Producer-consumer multiprocessing pattern. Constructor requires `system_id`, `data_logger`,
  `name`, and `output_directory` as positional arguments. `__init__` writes a camera manifest entry associating
  the system_id with the human-readable name. A producer process acquires frames from the camera interface and
  pushes them to a queue. A consumer process pops frames and streams raw bytes to FFMPEG via stdin. A 4-element
  SharedMemoryArray controls termination and frame-saving toggles via IPC.
- **GenICam Platform Support**: `harvesters` and `genicam` carry a
  `sys_platform != 'darwin' or (python_version < '3.14' and platform_machine == 'arm64')` marker, because `genicam`
  publishes a macOS wheel only for Apple Silicon on Python 3.12 and 3.13. Intel Macs and Macs on Python 3.14 therefore
  support no GenICam interface. `camera.py` mirrors that marker in `_GENICAM_RUNTIME_CLAIMED`, which selects the wording
  of `GENICAM_UNAVAILABLE_REASON`, and a guarded `try/except ImportError` keeps the module usable for the OpenCV and
  Mock interfaces there. `_require_genicam_runtime()` aborts every GenICam hardware entry point, and discovery reports
  the interface absent.
- **Camera Interfaces**: Three implementations behind a unified API: OpenCVCamera (cv2.VideoCapture),
  HarvestersCamera (GenICam/Harvesters with NodeMap access), and MockCamera (synthetic frames for testing).
  `discover_camera_ids()` returns CameraInformation objects from all available interfaces.
- **VideoSaver**: Manages an FFMPEG subprocess (Popen with stdin=PIPE). A daemon thread drains stderr continuously
  to prevent pipe buffer deadlocks. Supports CPU (libx264/libx265) and GPU (h264_nvenc/hevc_nvenc) encoders.
- **GenICam Configuration**: Iterative stack-based NodeMap traversal collects ReadWrite leaf nodes. Configurations
  serialize to YAML via GenicamConfiguration dataclass with camera identity metadata for validation on load.
  `read_camera_configuration()` is the library root's entry point for external callers, and `HarvestersCamera` and
  `harvester_connection` stay `video` package exports that the CLI and the MCP tools consume.
- **Camera Manifest**: `CameraManifest` (YamlConfig subclass) associates source IDs with human-readable camera
  names in a `camera_manifest.yaml` file alongside DataLogger archives. `CameraSourceData` stores per-camera
  entries. `write_camera_manifest()` creates or updates the manifest. Used by log processing discovery to identify
  which archives were produced by ataraxis-video-system and to route processing by source ID.
- **Log Processing**: Extracts frame acquisition timestamps from DataLogger `.npz` archives, one job per archive, into
  Feather files under a `camera_timestamps/` directory. `resolve_jobs()` reads the recording manifest, `prepare_jobs()`
  registers the resolved jobs without reading an archive, and `size_job()` resolves a job's width and memory from one
  archive read. `run_log_processing_pipeline()` runs one recording sequentially, or one job by canonical identifier.
- **MCP Server**: A shared `MCPServer` instance (`name="ataraxis-video-system"`) in `interfaces/mcp_instance.py` backs
  the tools in the `interfaces/*_tools.py` modules, and session globals in `interfaces/session_tools.py` enforce one
  active VideoSystem session. Batch log processing sizes each job with `size_job()` and admits what the core and memory
  budgets fit, running an oversized job alone. The ataraxis **video** plugin registers the server with MCP clients.
- **CLI**: Click command groups (`cti`, `check`, `configure`) with `run` for interactive sessions, `process` for
  log data processing, and `mcp` for starting the MCP server. CLI uses system_id 111, MCP uses 112.

### Key patterns

- **Multiprocessing Spawn**: `mp.set_start_method("spawn")` is set globally in `__init__.py` for cross-platform
  consistency. The VideoSystem's producer and consumer processes are daemon processes requiring explicit `stop()`
  calls, while the log processing pools spawn non-daemon workers that their owners shut down explicitly.
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
2. Static methods `_frame_production_loop()` and `_frame_saving_loop()` run in separate processes
3. Test with actual camera hardware or MockCamera interface

**Modifying camera interfaces:**

1. Review `src/ataraxis_video_system/video/camera.py` for all three implementations
2. OpenCVCamera and HarvestersCamera share a common interface pattern (connect, grab, disconnect)
3. CTI file management functions (`add_cti_file`, `check_cti_file`) use platformdirs for persistence
4. Camera discovery must handle both OpenCV and Harvesters gracefully when hardware is unavailable
5. Release the acquirer and the Harvester independently in `disconnect()`, since a failed `connect()` leaves a
   Harvester holding the GenTL Producer and stranding it hides every camera for the rest of the process

**Modifying FFMPEG encoding:**

1. Review `src/ataraxis_video_system/video/saver.py` for the VideoSaver class
2. Encoder command construction happens in `__init__()` based on VideoEncoders enum selection
3. CPU presets (veryfast-veryslow) map to GPU p1-p7 equivalents
4. The stderr drain thread is critical and must not be removed

**Modifying GenICam configuration:**

1. Review `src/ataraxis_video_system/video/configuration.py` for node traversal and serialization
2. Node enumeration uses iterative stack-based traversal, because a deep node tree overflows a recursive walk
3. GenicamConfiguration is a YamlConfig subclass supporting `to_yaml()` and `from_yaml()`
4. Strict identity checking compares camera model and serial number against YAML metadata

**Adding or modifying CLI commands:**

1. Review `src/ataraxis_video_system/interfaces/cli.py` for existing Click group structure
2. Use `console.echo()` for output and `console.error()` for error handling

**Modifying camera manifests:**

1. Review `src/ataraxis_video_system/video/manifest.py` for data classes and writer
2. `CameraManifest` extends `YamlConfig` from `ataraxis-data-structures` for YAML serialization
3. `CameraSourceData` is a frozen dataclass storing `id` (int) and `name` (str)
4. Manifests are consumed by `discover_camera_data_tool` in the MCP server for log processing discovery

**Modifying log processing:**

1. Review `orchestration/discovery.py` for resolution, preparation, and sizing, `orchestration/worker.py` for the
   single-job runner, and `video/timestamps.py` for the extraction algorithm
2. `extract_logged_camera_timestamps()` reads `.npz` archives via `LogArchiveReader` and returns `NDArray[np.uint64]`
3. `ProcessingTracker` manages job lifecycle (SCHEDULED → RUNNING → SUCCEEDED/FAILED) via YAML state files
4. `_process_frame_message_batch()` runs at the dispatch width of its job, in the pool workers that
   `extract_logged_camera_timestamps()` opens

**Adding or modifying MCP tools:**

1. Review the `interfaces/*_tools.py` modules for tool patterns (`interfaces/mcp_instance.py` holds the `mcp` instance)
2. Enforce single-session constraint via `_active_session` global state check
3. Log processing execution uses `JobExecutionState` (from `..orchestration`) with core and memory budgets
4. Most tools return `dict[str, Any]` structured data, and some return strings
