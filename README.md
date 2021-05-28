# sgx_dnn_benchmark

Simple benchmark aimed to compare performance of **cnn_training_f32** and **cnn_inference_f32** samples from the official package in simple CPU mode and in the enclave mode

Prerelease mode results obtained for the latest release of SGX SDK vs OneDnn 1.1.1 (on which the former is based) on Intel(R) Xeon(R) E-2288G CPU @ 3.70GHz are:

| Function                        |             Time, ms  | 
| ------------------------------- | --------------------- | 
| cnn_inference_f32_cpp on CPU    |            168.415345 |  
| cnn_training_f32_cpp on CPU     |            117.568182 |  
| cnn_inference_f32_cpp enclave   |           2701.513531 |
| cnn_training_f32_cpp enclave    |           3049.103561 |


# Prerequisites 
Make sure you have https://github.com/intel/linux-sgx/ and https://github.com/oneapi-src/oneDNN/tree/v1.1.1 installed.
Thus the benchmarking is fair since current versions of OneDnn may have gone way ahead of the fork used in SGX SDK (v1.1.1)

# Build
Use SGX SDK's parameters to build debug or release versions

for debug build:
```
make
```

for prerelease build:
```
make SGX_DEBUG=0 SGX_PRERELEASE=1
```

for release build (sign the enclave by yourself):
```
make SGX_DEBUG=0
```

# Run
Just launch the application produced by make

```
./app
```

# Benchmarking with OpenMP switched off
During the tests we found out that usage of OpenMP doesn't make any significant difference on performance. But since threading inside the enclave supposedly brings context switch / synchronization overhead, we find it reasonable to include the results with OpenMP turned off

# Build SGX SDK with OpenMP switched off

1. git clone https://github.com/intel/linux-sgx_dnnl

2. switch to the latest release tag

```
git checkout tags/sgx_2.13.3
```

2. Download dependencies (git submodule)

```
make preparation
```

3. Overwrite the file sgx_dnnl.patch in external/dnnl with the one provided

4. make sdk

5. make sdk_install_pkg

6. Install the sdk and input the desired installation dir
```
./linux/installer/bin/sgx_linux_x64_sdk_2.13.103.1.bin
source <install_dir>/sgxsdk/environment
```

7. Build Dnnl port for SGX 

```
cd external/dnnl
make
```

8. Put generated headers and library to the installation dir you selected in step 6.

```
cp external/dnnl/sgx_dnnl/include/* $SGX_SDK/include
cp external/dnnl/sgx_dnnl/lib/* $SGX_SDK/lib64
```

9. Make sure to remove all linkage to -lsgx_omp if you want to build existing sources targeting sgx_dnnl

# Benchmarking results with OpenMP turned off

on Intel(R) Xeon(R) E-2288G CPU @ 3.70GHz:

| Function                        | Time with OpenMp, ms  | Time without OpenMP, ms |
| ------------------------------- | --------------------- | ----------------------- |
| cnn_inference_f32_cpp on CPU    |            168.415345 |              584.704426 |
| cnn_training_f32_cpp on CPU     |            117.568182 |              237.126444 |
| cnn_inference_f32_cpp enclave   |           2701.513531 |             3992.817552 |
| cnn_training_f32_cpp enclave    |           3049.103561 |             3264.539662 |

on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz:

| Function                        | Time with OpenMp, ms  | Time without OpenMP, ms |
| ------------------------------- | --------------------- | ----------------------- |
| cnn_inference_f32_cpp on CPU    |            323.889681 |              365.733922 |
| cnn_training_f32_cpp on CPU     |            135.874076 |              135.197621 |
| cnn_inference_f32_cpp enclave   |             262.82887 |             144.6728862 |
| cnn_training_f32_cpp enclave    |            1795.22825 |              1921.58327 |

As we can see, while usage of OpenMP indeed makes difference for simple CPU mode, the usage of OpenMP in the enclave worsenes performance, making its usage inside the enclave iunreasonable
