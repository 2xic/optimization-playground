
gpu:
	nvcc gpu/gpu.cu -o gpu/gpu -I ./ ./gpu/kernel.cu -rdc=true

gpu_library:
	# https://forums.developer.nvidia.com/t/shared-library-creation/4776/4
	nvcc gpu/library.cu -o gpu/libgpu.so -I ./ ./gpu/kernel.cu -rdc=true --compiler-options '-fPIC --shared' --shared -std=c++11
	sudo cp gpu/libgpu.so /usr/lib/
	sudo cp cpu/libcpu.so /usr/local/lib/
	nm gpu/libgpu.so | grep test

gpu_run: gpu
	./gpu/gpu

cpu:
	gcc cpu/cpu.c -o cpu/cpu

cpu_library:
	gcc -shared -fPIC cpu/library.c cpu/shared.c -o cpu/libcpu.so 
	sudo cp cpu/libcpu.so /usr/lib/
	sudo cp cpu/libcpu.so /usr/local/lib/
	nm cpu/libcpu.so | grep test

test_build:
	gcc cpu/test_matmul.c -o cpu/test_matmul -I ./ ./cpu/shared.c 

test: test_build
	./cpu/test_matmul

python: cpu_library
	cd python-api && ./build.sh
	
.PHONY: gpu cpu test gpu_library cpu_library
