.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Video
=====

.. automodule:: ataraxis_video_system.video
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: ataraxis_video_system.video.manifest.CAMERA_MANIFEST_FILENAME

.. autodata:: ataraxis_video_system.video.configuration.DEFAULT_BLACKLISTED_NODES

.. autodata:: ataraxis_video_system.video.video_system.MAXIMUM_QUANTIZATION_VALUE

.. autodata:: ataraxis_video_system.video.camera.GENICAM_UNAVAILABLE_REASON

Orchestration
=============

.. automodule:: ataraxis_video_system.orchestration
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: ataraxis_video_system.orchestration.jobs.CAMERA_EXTRACTION_JOB_NAME

.. autodata:: ataraxis_video_system.orchestration.allocation.CAMERA_EXTRACTION_JOB_CORES

.. autodata:: ataraxis_video_system.orchestration.allocation._PARALLEL_EXTRACTION_THRESHOLD

.. autodata:: ataraxis_video_system.orchestration.allocation.SPAWNED_CHILD_MEMORY_MB

Command Line Interfaces (CLIs)
==============================

.. click:: ataraxis_video_system.interfaces.cli:axvs_cli
   :prog: axvs
   :nested: full
