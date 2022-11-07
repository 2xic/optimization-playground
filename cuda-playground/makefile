
gpu:
	nvcc gpu/gpu.cu -o gpu/gpu -I ./ ./gpu/kernel.cu -rdc=true

cpu:
	gcc cpu/cpu.c -o cpu/cpu

test_build:
	gcc cpu/test_matmul.c -o cpu/test_matmul -I ./ ./cpu/shared.c 

test: test_build
	./cpu/test_matmul

.PHONY: gpu cpu test
