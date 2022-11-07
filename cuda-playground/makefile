
gpu:
	nvcc gpu.cu -o gpu

cpu:
	gcc cpu.c -o cpu

test_build:
	gcc test_matmul.c -o test_matmul -I ./ ./shared.c 

test: test_build
	./test_matmul

.PHONY: gpu cpu test
