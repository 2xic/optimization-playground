python3 setup.py clean
#CC=nvcc python3 setup.py install
python3 setup.py install
echo "Trying to run :)"
python3 -c "import cudaplayground;"
