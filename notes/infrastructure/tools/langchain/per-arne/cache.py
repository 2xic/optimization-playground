import os
import hashlib

class Cache:
    def __init__(self, file) -> None:
        self.root = os.path.dirname(
            os.path.abspath(__file__)
        )
        self.hash = hashlib.sha256(
            file.read()
        ).hexdigest()
        file.seek(0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def cache(self, response):
        with open(self._cache_path, "w") as file:
            file.write(response)
        return response
    
    @property
    def cached(self):
        if os.path.isfile(
            self._cache_path
        ):
            with open(self._cache_path, "r") as file:
                return file.read()
        return False

    @property
    def _cache_path(self):
        path = os.path.join(
            self.root,
            ".cache"
        )
        os.makedirs(
            path,
            exist_ok=True
        )
        return os.path.join(
            path,
            self.hash
        )
    