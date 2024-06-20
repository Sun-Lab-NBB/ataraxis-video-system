"""This module provides a command-line interface for various project build and deployment tasks.

The methods exposed through this module are intended to be called through appropriate 'tox' pipelines and should not
be used directly.
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap
from os import PathLike
from typing import AnyStr, Optional
import yaml

import click


def format_message(message: str) -> str:
    """Quick local implementation of the standard lab utility method to format text messages."""
    return textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False)


def resolve_typed_markers(target_dir: AnyStr | PathLike[AnyStr]) -> None:
    """Crawls the input directory tree and resolves 'py.typed' marker file to match SunLab guidelines.

    Specifically, if the 'py.typed' is not found in the root directory, adds the marker file. If it is found in any
    subdirectory, removes the marker file.

    Note:
        The marker file has to be present in-addition to thy 'pyi' typing files to notify type-checkers, like mypy,
        that the library comes with type-annotations. This is necessary to allow other projects using type-checkers to
        verify this library API is accessed correctly.

    This subroutine is used to ensure typed marker is found only at the highest level of the library hierarchy.

    Args:
        target_dir: The path to the root level of the directory to crawl. Usually, this is the '/src' directory of the
        project.
    """
    for root, dirs, files in os.walk(target_dir):
        level: int = len(root.split(sep=os.path.sep))  # Tracks the evaluated directory level.

        # If evaluated directory is the root directory and the py.typed marker is not found, adds the marker file.
        if "py.typed" not in files and level == 1:
            # Add py.typed to this package directory
            with open(os.path.join(root, "py.typed"), "w") as _:
                pass

        # Removes any instances for all directories except the root directory.
        elif level > 1:
            dir_py_typed = os.path.join(root, "py.typed")
            if os.path.exists(dir_py_typed):
                os.remove(dir_py_typed)


def move_stubs(src_dir: str, dst_dir: str) -> None:
    """Moves typing stubs from the source folder to appropriate level(s) of the dist hierarchy.

    This procedure is intended to be executed after running stubgen on the compiled package instance, and it expects
    that the layout of the destination directory exactly matches the layout of the source directory.

    Args:
        src_dir: The path to the source directory (usually, this is the stubgen output directory).
        dst_dir: The path to the destination directory (usually, this is the '/src' directory of the project).
    """

    # Iterates over all files of the input tree hierarchy
    for root, _, files in os.walk(src_dir):
        for file in files:
            # For any file with python stub extension that matches the pattern, moves it to a mirroring directory level
            # and name relative to the destination root. Explicitly designed to filter out an occasional issue seen with
            # parallel tox runtimes, where an extra space_number is appended to the file.
            if file.endswith(".pyi") and not re.match(r".*\s\d+\.pyi$", file):
                stub_path = os.path.join(root, file)  # Parses the path to the stub file relative to the source root

                # Computes the would-be path of the file, if ti was saved inside the destination directory, rather than
                # the source. In other words, replaces the source directory with destination directory, leaving the
                # rest of the path intact
                split_path = stub_path.split(sep=os.path.sep)
                file_path = str(os.path.sep).join(split_path[2:])
                dst_path = os.path.join(dst_dir, file_path)

                # Removes old .pyi file if it already exists
                if os.path.exists(dst_path):
                    os.remove(dst_path)

                # Moves the stub to its destination directory
                shutil.move(stub_path, dst_path)


@click.group()
def cli() -> None:
    """This CLI exposes helper commands used to automate various project management steps. See below for details
    about the available commands.
    """
    pass


@cli.command()
def process_typed_markers() -> None:
    """Crawls the '/src' directory and ensures that the 'py.typed' marker is found only at the highest level of the
    library hierarchy (the highest directory with __init__.py in it).

    This command should be called as part of the stub-generation tox command.
    """
    if not os.path.exists("src"):
        click.echo("Unable to resolve typed markers. Source directory does not exist.", err=True)
        raise click.Abort()

    error_message: str = format_message(
        "Unable to resolve typed markers. Expected an __init__.py at the level of the src or a single "
        "sub-directory with an __init__.py, but discovered neither. C-extensions should use 'src' as the "
        "root directory and pure-python packages should have the library stored directly under src."
    )

    # If __init__.py is not found at the level of the src, this implies that the processed project is a pure python
    # project and, in this case, it is expected that there is a single library-directory under /src that is the
    # root.
    if "__init__.py" not in os.listdir("src"):
        if len(os.listdir("src")) > 1:
            click.echo(error_message, err=True)
            raise click.Abort()
        candidate_path: str = os.path.join("src", os.listdir("src")[0])
        if not os.path.isdir(candidate_path):
            click.echo(error_message, err=True)
            raise click.Abort()
        if "__init__.py" not in os.listdir(candidate_path):
            click.echo(error_message, err=True)
            raise click.Abort()
    # If __init__.py is found at the level of the src, this is used as a heuristic that implies that this library
    # is a c-extension library and does not contain a 'root' package (instead, src is the root).
    else:
        candidate_path: str = "src"

    # This is only executed if the conditional above was able to determine a good candidate_path.
    try:
        resolve_typed_markers(target_dir=candidate_path)
        click.echo("Typed Markers: Resolved.")
    except Exception as e:
        click.echo(f"Error resolving typed markers: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
def process_stubs() -> None:
    """Distributes the stub files from the '/stubs' directory to the appropriate level of the '/src' or
     'src/library' directory (depending on the type of the processed project).

    Notes:
        This command should only be called after the /stubs directory has been generated using stubgen command from tox.
    """
    if not os.path.exists("src"):
        click.echo("Unable to move stub files. Source directory does not exist.", err=True)
        raise click.Abort()
    if not os.path.exists("stubs"):
        click.echo("Unable to move stub files. Stubs directory does not exist.", err=True)
        raise click.Abort()

    # Resolves the candidate path for moving the stubs
    error_message: str = format_message(
        "Unable to resolve typed markers. Expected an __init__.py at the level of the src or a single "
        "sub-directory with an __init__.py, but discovered neither. C-extensions should use 'src' as the "
        "root directory and pure-python packages should have the library stored directly under src."
    )

    # If __init__.py is not found at the level of the src, this implies that the processed project is a pure python
    # project and, in this case, it is expected that there is a single library-directory under /src that is the
    # root.
    if "__init__.py" not in os.listdir("src"):
        if len(os.listdir("src")) > 1:
            click.echo(error_message, err=True)
            raise click.Abort()
        candidate_path: str = os.path.join("src", os.listdir("src")[0])
        if not os.path.isdir(candidate_path):
            click.echo(error_message, err=True)
            raise click.Abort()
        if "__init__.py" not in os.listdir(candidate_path):
            click.echo(error_message, err=True)
            raise click.Abort()

    # If __init__.py is found at the level of the src, this is used as a heuristic that implies that this library
    # is a c-extension library and does not contain a 'root' package (instead, src is the root).
    else:
        candidate_path: str = "src"

    try:
        move_stubs(src_dir="stubs", dst_dir=candidate_path)  # Distributes the stubs across source directory
        shutil.rmtree("stubs")  # Removes the directory
        click.echo("Stubs: Distributed.")
    except Exception as e:
        click.echo(f"Error processing stubs: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
def generate_recipe_folder() -> None:
    """Generates the recipe folder used by Grayskull.

    Since Grayskull does not generate output folders, this has to be outsourced to a separate command.
    """
    if not os.path.exists("recipe"):
        try:
            os.makedirs("recipe")
            click.echo("Recipe Directory: Generated.")
        except Exception as e:
            click.echo(f"Error generating recipe folder: {str(e)}", err=True)
            raise click.Abort()
    else:
        shutil.rmtree("recipe")
        os.makedirs("recipe")
        click.echo("Recipe Directory: Recreated.")


@cli.command()
@click.option(
    "--replace-token",
    "-r",
    type=click.BOOL,
    default=False,
    help="If provided, this flag forces the method to replace the API token stored in the .pypirc file with a new one.",
)
def set_pypi_token(replace_token: bool) -> None:
    """Ensures the .pypirc file exists at the root of the project and stores the user API token to upload compiled
    package to PIP.

    The '.pypirc' file is added to gitignore, so there should be no private information leaking unless
    gitignore is not included.
    """
    if not os.path.exists(".pypirc") or replace_token:
        try:
            token = click.prompt(
                "Enter your PyPI (API) token. It will be stored inside the .pypirc file for future use:",
                hide_input=True,
            )
            with open(".pypirc", "w") as f:
                f.write(f"[pypi]\nusername = __token__\npassword = {token}\n")
            click.echo("PyPI Token: Added to '.pypirc'.")
        except Exception as e:
            click.echo(f"Error setting PyPI token: {str(e)}", err=True)
            raise click.Abort()
    else:
        click.echo("PyPI Token: '.pypirc' file already exists. Use the '--replace-token' flag to update the token.")


def get_env_extension(os_name: str) -> str:
    """Returns the postfix used to identify the Conda development environment for each of the 3 major supported
    operational systems: OSx (Apple Silicon), Linux and Windows.
    """
    if os_name == "win32":
        return "_win64"
    elif os_name == "linux":
        return "_lin64"
    elif os_name == "darwin":
        return "_osx"
    else:
        raise click.BadParameter(f"Unsupported operating system: {os_name}.")


@cli.command()
def import_env() -> None:
    """Creates or updates an existing Conda environment based on the operating system-specific .yml file."""
    os_name: str = sys.platform  # Gets the host os name
    env_postfix: str = get_env_extension(os_name)  # Uses os name to generate the appropriate environment postfix
    yml_file: str = f"{env_postfix}.yml"  # CConcatenates the postfix with the .yml extension

    # Scans the 'envs' directory and discovers the first file with the matching postfix and extension. If a match is
    # found, uses it to set the path to the .yml file and the name to use in conda
    yml_path: Optional[str] = None
    env_name: Optional[str] = None
    with os.scandir("envs") as iterator:
        for file in iterator:
            if yml_file in file.name:
                yml_path = os.path.join("envs", file.name)
                env_name = file.name.split(".")[0]
                break

    # If the os-specific .yml file is not found, raises an error
    if not os.path.exists(yml_path):
        click.echo(
            f"No environment file found for the requested postfix and extension combination {yml_file}", err=True
        )
        raise click.Abort()

    # If the .yml file was found, attempts to create a new environment by calling Conda from subprocess
    try:
        subprocess.run(["conda", "env", "create", "-f", yml_path], check=True)
        click.echo(f"Environment '{env_name}' created successfully.")
    except Exception:
        # If environment creation fails, this is likely due to the environment already existing. Therefore, upon the
        # first error, attempts to instead update the existing environment using the same.yml file
        try:
            subprocess.run(["conda", "env", "update", "-f", yml_path], check=True)
            click.echo(f"Environment '{env_name}' already exists and was instead updated successfully.")
        # If the update attempt also fails, aborts with an error.
        except subprocess.CalledProcessError as e:
            click.echo(f"Unable to create or update an environment: {str(e)}", err=True)
            raise click.Abort()


@cli.command()
@click.option(
    "--base-env",
    prompt="Enter the base environment name",
    required=True,
    help="The base name of the environment to export.",
)
def export_env(base_env: str) -> None:
    """Exports the os-specific Conda environment as a .yml file and a spec-file .txt.

    Args:
        base_env: The base name (e.g.: axt_dev) of the environment. The os-specific postfix for the environment is
            resolved and appended automatically.
    """

    # Selects the environment name according to the host os.
    os_name: str = sys.platform
    env_extension: str = get_env_extension(os_name)
    env_name: str = f"{base_env}{env_extension}"

    try:
        # Exports environment as .yml file
        yml_file: str = f"{env_name}.yml"
        yml_path: str = os.path.join("envs", yml_file)

        # The .yml export method uses an os-specific procedure to remove the environment prefix from the exported file
        # as a privacy measure.
        if "_lin64" in env_extension:
            subprocess.run(f"conda env export --name {env_name} | head -n -1 > {yml_path}", shell=True, check=True)
        elif "_osx" in env_extension:
            subprocess.run(
                f"conda env export --name {env_name} | tail -r | tail -n +2 | tail -r > {yml_path}",
                shell=True,
                check=True,
            )
        elif "_win64" in env_extension:
            subprocess.run(
                f'conda env export --name {env_name} | findstr -v "prefix" > {yml_path}', shell=True, check=True
            )
        else:
            raise click.BadParameter(f"Unsupported operating system: {os_name}")

        click.echo(f"Environment exported to {yml_path}.")

        # Exports environment as spec-file .txt
        spec_file: str = f"{env_name}_spec.txt"
        spec_path: str = os.path.join("envs", spec_file)
        subprocess.run(f"conda list -n {env_name} --explicit -r > {spec_path}", shell=True, check=True)
        click.echo(f"Environment spec-file exported to {spec_path}")

    except subprocess.CalledProcessError as e:
        click.echo(f"Error exporting environment: {str(e)}", err=True)
        raise click.Abort()


def validate_library_name(_ctx, _param, value: str) -> str:
    """Verifies that the input library name contains only letters, numbers, and underscores."""
    if not re.match(r"^[a-zA-Z0-9_]*$", value):
        raise click.BadParameter("Library name should contain only letters, numbers, and underscores.")
    return value


def validate_project_name(_ctx, _param, value: str) -> str:
    """Verifies that the input project name contains only letters, numbers, and dashes."""
    if not re.match(r"^[a-zA-Z0-9-]+$", value):
        raise click.BadParameter("Project name should contain only letters, numbers, or dashes.")
    return value


def validate_author_name(_ctx, _param, value: str) -> str:
    """Verifies that the input author name contains only letters, numbers, spaces, underscores and dashes."""
    if not re.match(r"^[a-zA-Z0-9\s_-]+$", value):
        raise click.BadParameter("Author name should contain only letters, numbers, spaces, underscores, or dashes.")
    return value


def validate_email(_ctx, _param, value: str) -> str:
    """Verifies that the input email address contains only valid characters."""
    if not re.match(r"^[\w.-]+@[\w.-]+\.\w+$", value):
        raise click.BadParameter("Invalid email address.")
    return value


def validate_env_name(_ctx, _param, value: str) -> str:
    """Verifies that the input environment name contains only letters, numbers, and underscores."""
    if not re.match(r"^[a-zA-Z0-9_]*$", value):
        raise click.BadParameter("Environment name should contain only letters, numbers, and underscores.")
    return value


@cli.command()
@click.option("--library-name", prompt="Enter the desired library name", callback=validate_library_name)
@click.option("--project-name", prompt="Enter the desired project name", callback=validate_project_name)
@click.option("--author-name", prompt="Enter the author name", callback=validate_author_name)
@click.option("--email", prompt="Enter the email address", callback=validate_email)
@click.option("--env-name", prompt="Enter the environment name", callback=validate_env_name)
def adopt_project(library_name: str, project_name: str, author_name: str, email: str, env_name: str) -> None:
    """Adopt a new project that was initialized from a standard Sun Lab template, by replacing placeholders in metadata
    and automation files with user0-defined data.

    Specifically, it is used to replace predefined placeholders left throughout project automation and metadata files
    inside the template repositories with user-defined values, expediting initial project setup. At this time, the
    function is used to set: project name, library name, development (conda) environment BASE name, author name and
    author email. In the future, more markers may be added as needed.

    Note:
        Manual validation of all automation files is highly advised. This function is not intended to replace manual
        configuration, but only to expedite it in critical bottlenecks. It is very likely that your project will not
        work as expected without additional configuration.

    Args:
        library_name: The name of the library. This is what the end-users will 'import' when they use the library.
        project_name: The name of the project. This is what the end-users will 'pip install'.
        author_name: The name of the author. If more than one author works on the library, you will need to manually
            add further authors through pyproject.toml.
        email: The email address of the author. For multiple authors, see above.
        env_name: The base name to use for the conda (or mamba) environment used by the project. This primarily controls
            the name automatically given to all exported conda files (via export-env automation command).

    """
    # Stores the placeholder markers alongside their replacement values
    markers = {
        "YOUR_LIBRARY_NAME": library_name,
        "YOUR-PROJECT-NAME": project_name,
        "YOUR_AUTHOR_NAME" : author_name,
        "YOUR_EMAIL"       : email,
        "YOUR_ENV_NAME"    : env_name,
        "template_ext"     : env_name,
        "template_pure"    : env_name,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exclude_file_names = ["automation.py"]

    # Uses the input environment name to rename all environment files inside the 'envs' folder.
    rename_all_envs(env_name)

    try:
        # Loops over the script directory, which should be project root directory
        total_markers: int = 0  # Tracks the number of replaced markers.
        for root, dirs, files in os.walk(script_dir):
            # Discovers the target files to be modified
            for file in files:
                file_path = os.path.join(root, file)
                _, file_ext = os.path.splitext(file)

                # Opens and reads the contents of the files ot be modified
                if file not in exclude_file_names:

                    # Processes file names
                    if file in markers.items():
                        os.rename(src=file_path, dst=os.path.join(root, file))

                    # Processes file contents
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Loops over markers and if any are discovered inside the evaluated file contents, replaces the
                    # markers with the corresponding value. If the marker is not found, the file is not modified.
                    content_modified = False
                    for marker, value in markers.items():
                        if marker in content:
                            content = content.replace(marker, value)
                            total_markers += 1
                            content_modified = True

                    if content_modified:
                        with open(file_path, "w") as f:
                            f.write(content)
                        click.echo(f"Replaced markers in {file_path}")

        # Provides the final reminder
        message = (
            f"Project Adoption: Complete. Be sure to manually verify critical files such as CMakeLists.txt and "
            f"pyproject.toml before proceeding to the next step. Overall, found and replaced {total_markers} markers."
        )
        click.echo(format_message(message))
    except Exception as e:
        click.echo(f"Error replacing markers: {str(e)}", err=True)
        raise click.Abort()


def rename_all_envs(new_name: str) -> None:
    """This task loops over all files inside the '/envs' directory, replaces base environment names with the input
    new name, and updates the environment names inside the .yml files.

    It is mainly designed to be used during template project adoption, but also can be used as part of tox-automation to
    rename all environments in the folder (for example, when changing the environment naming pattern for the project).

    Args:
        new_name: The new base name to use for all environments.
    """

    envs_dir: str = "envs"
    if os.path.exists(envs_dir):
        for file in os.listdir(envs_dir):
            if file.endswith(".yml"):
                last_underscore_index = file.rfind("_")
                if last_underscore_index != -1:
                    os_suffix_and_ext = file[last_underscore_index:]
                    new_file_name = f"{new_name}{os_suffix_and_ext}"  # Underscore from suffix is kept
                    old_file_path = os.path.join(envs_dir, file)
                    new_file_path = os.path.join(envs_dir, new_file_name)

                    # Read the YAML file
                    with open(old_file_path, "r") as f:
                        yaml_data = yaml.safe_load(f)

                    # Update the environment name inside the YAML file
                    if "name" in yaml_data:
                        yaml_data["name"] = new_file_name[:-4]  # Remove the '.yml' extension

                    # Write the updated YAML data to the new file
                    with open(new_file_path, "w") as f:
                        yaml.safe_dump(yaml_data, f)

                    # Remove the old file
                    os.remove(old_file_path)

                    click.echo(f"Renamed environment file: {file} -> {new_file_name}")
            elif file.endswith("_spec.txt"):
                # finds the first underscore starting from _spec.txt (excludes the spec underscore)
                last_underscore_index = file.rfind("_", 0, file.rfind("_spec.txt"))
                if last_underscore_index != -1:
                    os_suffix_and_ext = file[last_underscore_index:]
                    new_file_name = f"{new_name}{os_suffix_and_ext}"
                    old_file_path = os.path.join(envs_dir, file)
                    new_file_path = os.path.join(envs_dir, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    click.echo(f"Renamed environment file: {file} -> {new_file_name}")


@cli.command()
@click.option("--new-name", prompt="Enter the new base environment name to use:", callback=validate_env_name)
def rename_environments(new_name: str) -> None:
    # This is basically the wrapper for the shared method.
    rename_all_envs(new_name=new_name)


if __name__ == "__main__":
    cli()
