# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Code Contributions and Review

Before writing, modifying, or reviewing any code, you MUST invoke the `/sun-lab-style` skill to load the Sun Lab
Python coding conventions. All code contributions must strictly follow these conventions and all code reviews must
check for compliance. Key conventions include:
- Google-style docstrings with proper sections
- Full type annotations with explicit array dtypes
- Keyword arguments for function calls
- Third person imperative comments
- Proper error handling with `console.error()`

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored
locally in the same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g.,
   `../ataraxis-time/`, `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or
   main branch on GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch,
   notify the user with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative
   reference for API usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the
actual library state to prevent integration errors.

## MCP Server Integration

Sun Lab libraries may provide MCP (Model Context Protocol) servers that expose library functionality for agentic
interaction. When working with this project or its dependencies, prefer using available MCP tools over direct code
execution when appropriate.

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools
   they provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like camera discovery, video session management, and
   runtime checks, use MCP tools rather than writing and executing Python code directly. MCP tools provide:
   - Consistent, tested interfaces
   - Proper resource management and cleanup
   - Formatted output designed for user display

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when:
   - No MCP tool exists for the required functionality
   - The task requires custom logic not covered by available tools
   - Writing or modifying library source code

**Example**: Instead of writing Python code to discover cameras, use the `list_cameras` MCP tool if available.

## Available Skills

- `/explore-codebase` - Perform in-depth codebase exploration
- `/sun-lab-style` - Apply Sun Lab Python coding conventions (REQUIRED for all code changes)

## Project Context

This is **ataraxis-video-system**, a Python library for interfacing with a wide range of cameras to flexibly record
visual stream data as video files. The library supports both OpenCV-compatible cameras and GeniCam-compliant
industrial cameras via the Harvesters library.

### Key Areas

| Directory                                  | Purpose                                          |
|--------------------------------------------|--------------------------------------------------|
| `src/ataraxis_video_system/`               | Main library source code                         |
| `src/ataraxis_video_system/cli.py`         | CLI entry point (`axvs` command)                 |
| `src/ataraxis_video_system/mcp_server.py`  | MCP server for agentic integration               |
| `examples/`                                | Usage examples and demonstrations                |
| `tests/`                                   | Test suite                                       |

### Architecture

- CLI interface for camera discovery and management (`axvs id`, `axvs cti`, `axvs cti-check`, `axvs check`, `axvs run`)
- MCP server for AI agent integration (`axvs mcp`)
- Unified camera abstraction supporting multiple backends (OpenCV, Harvesters/GeniCam)
- Video recording with FFMPEG-based encoding
- Timestamp synchronization for scientific applications

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `/sun-lab-style` for complete conventions
