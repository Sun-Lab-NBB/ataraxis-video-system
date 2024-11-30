from .video_system import VideoSystem as VideoSystem

def list_ids(backend: str, cti_path: str) -> None:
    """Lists ids for the cameras available through the selected interface. Subsequently, the IDs from this list can be
    used when instantiating Camera class through API or as ain input to 'live-run' CLI script.

    This method is primarily intended to be used on systems where the exact camera layout is not known. This is
    especially true for the OpenCV id-discovery, which does not provide enough information to identify cameras.
    """
