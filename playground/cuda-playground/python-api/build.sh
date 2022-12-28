#python3 setup.py clean
#CC=nvcc python3 setup.py install
python3 setup.py build --force
python3 setup.py install --force
echo "Trying to run :)"
python3 -c "import cudaplayground;"
