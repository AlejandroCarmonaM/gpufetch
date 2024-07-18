# gpufetch

![gpufetch logo](./logo.png)

gpufetch is a command-line tool that provides detailed information about NVIDIA GPU(s) installed on your system. It is designed to be a introduction to how CUDA programmers can use GPU queried information to better its code. Besides a reduced set of information smaller than cuda-samples deviceQuery.cpp, it also introduces Tensor Core information based on the GPU Architecture and performs a Simple Memory Benchmark for Global Memory and L2 using queried information  

## Features

* **Device Information**: Retrieve detailed specifications of installed NVIDIA GPUs including device name, clock rate, memory bus width, and peak memory bandwidth.
* **Memory Details**: Display total global memory, L2 cache size, shared memory per block, and registers per block.
* **Compute Capabilities**: View compute capability, maximum threads per SM, and warp size.
* **Tensor Core Information**: Information on Tensor Cores, including architecture, number of Tensor Cores per SM, and total number of Tensor Cores.
* **Performance Metrics**: Peak performance specifications for CUDA Cores in FP16, FP32, and FP64.
* **Benchmark Results**: Perform and display simple memory benchmarks for L2 cache bandwidth and global memory bandwidth.
* **Configuration Tips**: Includes notes on configuring L1 cache and shared memory using CUDA functions.




## Installation

1. Clone the repository:

```shell
git clone https://github.com/username/gpufetch.git
```

2. Navigate to the project directory:

```shell
cd gpufetch
```

3. Build the project:

```shell
make
```

4. Run gpufetch:

```shell
./gpufetch
```

## Usage

To display information about the GPU(s) on your system, simply run the `gpufetch` command:

```shell
./gpufetch
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
