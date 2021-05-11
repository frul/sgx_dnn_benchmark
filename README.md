# sgx_dnn_benchmark

# Prerequisites 
Make sure you have https://github.com/intel/linux-sgx and https://github.com/oneapi-src/oneDNN installed 


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
