# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load Sun
Lab conventions. This applies to ALL file types:

| Task                                | Skill to Invoke    |
|-------------------------------------|--------------------|
| Writing or modifying Python code    | `/python-style`    |
| Writing or modifying README files   | `/readme-style`    |
| Writing git commit messages         | `/commit`          |
| Writing or modifying pyproject.toml | `/pyproject-style` |
| Configuring tox.ini                 | `/tox-config`      |

All contributions must strictly follow these conventions. Key conventions include:
- Google-style docstrings with proper sections
- Full type annotations with explicit array dtypes
- Keyword arguments for function calls
- Third person imperative mood for comments and documentation
- Proper error handling with `console.error()`
- 120 character line limit

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

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

## Distribution Model

This project follows a dual distribution model. The library source code, tests, CLI, and MCP server implementation live
in this repository (`ataraxis-video-system`) and are distributed via PyPI. Claude Code skills and MCP server
registration are distributed separately through the [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as
plugins:

- **video** plugin (`ataraxis/plugins/video/`): Registers the `axvs mcp` server with compatible MCP clients and
  provides video-specific skills for camera setup, pipeline orchestration, log processing, and post-recording
  verification.
- **automation** plugin (`ataraxis/plugins/automation/`): Provides shared development skills that enforce Sun Lab
  coding conventions (Python style, README style, commit messages, pyproject.toml, tox configuration) and
  general-purpose codebase exploration tools.

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository, not in this repository.
When modifying the MCP server implementation or library code, edit the source files in this repository.

## MCP Server Integration

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

## Available Skills

Skills are distributed through the ataraxis marketplace and are loaded into Claude Code via the plugin system. They are
**not** stored in this repository.

### Video Plugin Skills (ataraxis/plugins/video/)

| Skill                     | Description                                                          |
|---------------------------|----------------------------------------------------------------------|
| `/camera-setup`           | MCP-based camera discovery, testing, encoding guidance, and GenICam  |
| `/camera-interface`       | VideoSystem API usage, constructor parameters, and encoding guidance |
| `/mcp-environment-setup`  | MCP server connectivity diagnostics and environment verification     |
| `/post-recording`         | Post-recording verification: log assembly, video validation, handoff |
| `/pipeline`               | End-to-end pipeline orchestration and multi-camera planning          |
| `/log-input-format`       | Reference for NPZ archive format, source IDs, and DataLogger output  |
| `/log-processing`         | Orchestrate log archive processing workflow via MCP tools            |
| `/log-processing-results` | Reference for output data formats and frame statistics analysis      |

### Automation Plugin Skills (ataraxis/plugins/automation/)

| Skill                     | Description                                                          |
|---------------------------|----------------------------------------------------------------------|
| `/explore-codebase`       | Perform in-depth codebase exploration at session start               |
| `/explore-dependencies`   | Explore installed ataraxis dependency APIs for reuse opportunities   |
| `/python-style`           | Apply Sun Lab Python coding conventions (REQUIRED for code changes)  |
| `/readme-style`           | Apply Sun Lab README conventions                                     |
| `/commit`                 | Draft Sun Lab style-compliant git commit messages                    |
| `/pyproject-style`        | Apply Sun Lab pyproject.toml conventions                             |
| `/tox-config`             | Apply Sun Lab tox.ini conventions                                    |
| `/skill-design`           | Generate and verify Claude Code skill files                          |
| `/project-layout`         | Apply Sun Lab project directory structure conventions                |

## Project Context

This is **ataraxis-video-system**, a Python library that interfaces with a wide range of cameras to flexibly record
visual stream data as video files. It supports OpenCV and GeniCam (Harvesters) camera interfaces with FFMPEG-based
video encoding using CPU or GPU.

### Key Areas

| Directory                    | Purpose                                                                  |
|------------------------------|--------------------------------------------------------------------------|
| `src/ataraxis_video_system/` | Main library source code                                                 |
| `src/.../video_system.py`    | Core VideoSystem class with multiprocessing architecture                 |
| `src/.../camera.py`          | Camera interfaces (OpenCV, Harvesters, Mock) and CTI management          |
| `src/.../saver.py`           | VideoSaver with FFMPEG subprocess encoding                               |
| `src/.../configuration.py`   | GenICam node inspection, read/write, dump/load via YAML                  |
| `src/.../manifest.py`        | Camera log manifest data classes and writer for source-to-name mappings  |
| `src/.../log_processing.py`  | Log data processing pipeline for extracting frame timestamps             |
| `src/.../cli.py`             | Click-based `axvs` CLI with subcommand groups                            |
| `src/.../mcp_server.py`      | FastMCP server with 27 tools for camera, session, and log management     |
| `tests/`                     | Test suite (camera, saver, video_system, configuration, log_processing)  |
| `docs/`                      | Sphinx API documentation source                                          |

### Architecture

- **VideoSystem**: Producer-consumer multiprocessing pattern. Constructor requires `system_id`, `data_logger`,
  `name`, and `output_directory` as positional arguments; `__init__` writes a camera manifest entry associating
  the system_id with the human-readable name. A producer process acquires frames from the camera interface and
  pushes them to a queue. A consumer process pops frames and streams raw bytes to FFMPEG via stdin. A 4-element
  SharedMemoryArray controls termination and frame-saving toggles via IPC.
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
- **Log Processing**: Pipeline for extracting frame acquisition timestamps from DataLogger `.npz` archives.
  Requires a `camera_manifest.yaml` in the log directory for source ID resolution and validation. Supports
  sequential and parallel (ProcessPoolExecutor) processing with contiguous numpy array output to minimize
  memory footprint. Uses `LogArchiveReader` for archive access and `ProcessingTracker` for job lifecycle
  management. `run_log_processing_pipeline()` orchestrates local (all jobs) and remote (single job by ID)
  execution modes. Outputs Polars DataFrames as Feather files in a `camera_timestamps/` subdirectory.
- **MCP Server**: FastMCP instance (`name="ataraxis-video-system"`, `json_response=True`) with 27 tools and
  global state (`_active_session`, `_active_logger`, `_session_info`) enforcing a single active VideoSystem
  session at a time. Tool categories: camera discovery and CTI management (3), system checks (1), video session
  lifecycle (5), GenICam configuration (4), camera manifest management (2), log archive and video validation (2),
  recording discovery via manifests (1), batch log processing execution (3), processing status and management (4),
  and post-processing analysis and cleanup (2). Session tools expose configurable encoding parameters (encoder,
  speed preset, pixel format, quantization). `stop_video_session` auto-assembles log archives and returns output
  paths. Batch log processing uses `_JobExecutionState` with budget-based worker allocation: the execution manager
  divides the CPU budget evenly among concurrent parallel jobs (snapped to multiples of 5) with a sqrt-derived
  saturation floor. The MCP server is registered with MCP clients via the **video** plugin in the ataraxis
  marketplace, not directly from this repository.
- **CLI**: Click command groups (`cti`, `check`, `configure`) with `run` for interactive sessions, `process` for
  log data processing, and `mcp` for starting the MCP server. CLI uses system_id 111, MCP uses 112.

### Key Patterns

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

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow Guidance

**Modifying VideoSystem:**

1. Review `src/ataraxis_video_system/video_system.py` for current implementation
2. Understand the producer-consumer multiprocessing architecture and SharedMemoryArray IPC
3. Static methods `_frame_production_loop()` and `_frame_saving_loop()` run in separate processes
4. Test with actual camera hardware or MockCamera interface

**Modifying camera interfaces:**

1. Review `src/ataraxis_video_system/camera.py` for all three implementations
2. OpenCVCamera and HarvestersCamera share a common interface pattern (connect, grab, disconnect)
3. CTI file management functions (`add_cti_file`, `check_cti_file`) use platformdirs for persistence
4. Camera discovery must handle both OpenCV and Harvesters gracefully when hardware is unavailable

**Modifying FFMPEG encoding:**

1. Review `src/ataraxis_video_system/saver.py` for the VideoSaver class
2. Encoder command construction happens in `__init__()` based on VideoEncoders enum selection
3. CPU presets (veryfast-veryslow) map to GPU p1-p7 equivalents
4. The stderr drain thread is critical and must not be removed

**Modifying GenICam configuration:**

1. Review `src/ataraxis_video_system/configuration.py` for node traversal and serialization
2. Node enumeration uses iterative stack-based traversal (not recursive)
3. GenicamConfiguration is a YamlConfig subclass supporting `to_yaml()` and `from_yaml()`
4. Strict identity checking compares camera model and serial number against YAML metadata

**Adding or modifying CLI commands:**

1. Review `src/ataraxis_video_system/cli.py` for existing Click group structure
2. Follow existing patterns for option decorators and error handling
3. Use `console.echo()` for output and `console.error()` for error handling

**Modifying camera manifests:**

1. Review `src/ataraxis_video_system/manifest.py` for data classes and writer
2. `CameraManifest` extends `YamlConfig` from `ataraxis-data-structures` for YAML serialization
3. `CameraSourceData` is a frozen dataclass storing `id` (int) and `name` (str)
4. Manifests are consumed by `discover_camera_data_tool` in the MCP server for log processing discovery

**Modifying log processing:**

1. Review `src/ataraxis_video_system/log_processing.py` for the processing pipeline
2. `extract_logged_camera_timestamps()` reads `.npz` archives via `LogArchiveReader` from `ataraxis-data-structures`
3. `run_log_processing_pipeline()` supports local mode (all jobs sequentially) and remote mode (single job by ID)
4. `ProcessingTracker` manages job lifecycle (SCHEDULED → RUNNING → SUCCEEDED/FAILED) via YAML state files
5. `extract_logged_camera_timestamps()` returns `NDArray[np.uint64]` for minimal memory footprint
6. `_process_frame_message_batch()` runs in subprocess workers and is excluded from coverage (`# pragma: no cover`)
7. Log discovery uses manifest-based routing via `camera_manifest.yaml` files

**Adding or modifying MCP tools:**

1. Review `src/ataraxis_video_system/mcp_server.py` for existing tool patterns
2. Enforce single-session constraint via `_active_session` global state check
3. Log processing execution uses `_JobExecutionState` with budget-based worker allocation
4. The execution manager divides budget among parallel jobs via `_resolve_parallel_allocation()`
5. Return formatted strings (not raw data) for user-facing output
6. MCP server registration happens in the ataraxis marketplace video plugin, not in this repository
