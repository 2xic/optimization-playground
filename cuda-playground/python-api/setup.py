from distutils.core import setup, Extension


def main():
    setup(name="cudaplayground",
          version="1.0.0",
          description="Python interface for the cudaplayground",
          author="2xic",
          author_email="me@2xic.xyz",
          ext_modules=[Extension(
              "cudaplayground",
              ["cuda_playground_module.c"], #, "tensor.c"],
              library_dirs=['/usr/lib/'],
              libraries=['gpu', 'cpu'],
          )])


if __name__ == "__main__":
    main()
    