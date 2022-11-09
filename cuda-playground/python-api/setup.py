from distutils.core import setup, Extension

def main():
    setup(name="cudaplayground",
          version="1.0.0",
          description="Python interface for the cudaplayground",
          author="2xic",
          author_email="me@2xic.xyz",
          ext_modules=[Extension(
            "cudaplayground", 
            ["cudaplaygroundmodule.c"],
            libraries=['/home/brage/Desktop/paper-zoo/cuda-playground/python-api/gpu']
          )])

if __name__ == "__main__":
    main()
