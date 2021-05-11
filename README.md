# sgx_dnn_benchmark

Simple benchmark aimed to compare performance of **cnn_training_f32** and **cnn_inference_f32** samples from the official package in simple CPU mode and in the enclave mode

Prerelease mode results obtained for the latest release of SGX SDK vs OneDnn 1.1.1 (on which the former is based) on Intel(R) Xeon(R) E-2288G CPU @ 3.70GHz are:

```
Inference on CPU time is: 168415345 [nanoseconds] 
Training on CPU time is: 117568182 [nanoseconds] 
Inference in enclave time is: 2701513531 [nanoseconds] 
Training in enclave time is: 3049103561 [nanoseconds] 
```

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
