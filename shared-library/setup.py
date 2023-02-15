import setuptools

print(setuptools.find_packages(where="src"))

setuptools.setup(
    name = "optimization_playground_shared",
    version = "0.0.1",
    author = "2xic",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
)
