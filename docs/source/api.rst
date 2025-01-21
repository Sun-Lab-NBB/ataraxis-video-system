.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (autodoc, etc.).

Video System
============

.. automodule:: ataraxis_video_system.video_system
   :members:
   :undoc-members:
   :show-inheritance:

Cameras
=======

.. automodule:: ataraxis_video_system.camera
   :members:
   :undoc-members:
   :show-inheritance:

Savers
======

.. automodule:: ataraxis_video_system.saver
   :members:
   :undoc-members:
   :show-inheritance:

Live Run CLI
============

.. click:: ataraxis_video_system.live:live_run
   :prog: axvs-live
   :nested: full

List Camera IDs CLI
===================

.. click:: ataraxis_video_system.list_camera_ids:list_ids
   :prog: axvs-ids
   :nested: full
