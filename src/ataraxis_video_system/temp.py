import tempfile
import os


with tempfile.TemporaryDirectory() as temp_dir:
    metadata_path = os.path.join(temp_dir, 'test_metadata')
    os.makedirs(metadata_path, exist_ok=True)
    print(metadata_path)
    input()
