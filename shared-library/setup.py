import setuptools
import os 

path = os.path.join(
    os.path.dirname(__file__),
    "requirements.txt"
)
with open(path) as f:
    required = f.read().splitlines()

print(setuptools.find_packages(where="src"))

setuptools.setup(
    name = "optimization_playground_shared",
    version = "0.0.1",
    author = "2xic",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    install_requires=required,
)
