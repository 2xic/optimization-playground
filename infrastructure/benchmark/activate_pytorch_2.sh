echo "test"

if [ ! -d "./pytorch_2_0" ] 
then
    python3 -m venv ./pytorch_2_0
fi
source pytorch_2_0/bin/activate

python3 -m pip install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117

python3 jit_vs_non_jit_speed.py
