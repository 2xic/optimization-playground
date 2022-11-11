from distutils.core import setup, Extension

def main():
    setup(name="cudaplayground",
          version="1.0.0",
          description="Python interface for the cudaplayground",
          author="2xic",
          author_email="me@2xic.xyz",
          package_data={
            'libcpu.so': ['libcpu.so']
          },
          ext_modules=[Extension(
            "cudaplayground", 
            ["cudaplaygroundmodule.c"],
#            libraries=['/home/brage/Desktop/paper-zoo/cuda-playground/python-api/gpu']
#            library_dirs=['/home/brage/Desktop/paper-zoo/cuda-playground/python-api/cpu/', './'],
            libraries=['gpu'],
          )])
    """
      P.s also move the package to /usr/lib
    """

if __name__ == "__main__":
    main()
