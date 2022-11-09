
gpu:
	nvcc gpu/gpu.cu -o gpu/gpu -I ./ ./gpu/kernel.cu -rdc=true

gpu_library:
	# https://forums.developer.nvidia.com/t/shared-library-creation/4776/4
	nvcc gpu/library.cu -o gpu/gpu.so -I ./ ./gpu/kernel.cu -rdc=true --compiler-options '-fPIC' --shared 

gpu_run: gpu
	./gpu/gpu

cpu:
	gcc cpu/cpu.c -o cpu/cpu

test_build:
	gcc cpu/test_matmul.c -o cpu/test_matmul -I ./ ./cpu/shared.c 

test: test_build
	./cpu/test_matmul

.PHONY: gpu cpu test 