#include <string.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <chrono>

#include "sgx_urts.h"
#include "sgx_tseal.h"

#include "../common/common.hpp"
#include "Enclave_u.h"
#include "Routine.h"
#include "BenchmarkRoutine.h"

#define ENCLAVE_NAME "libenclave.signed.so"

// Global data
sgx_enclave_id_t global_eid = 0;

typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug; /* Suggestion */
} sgx_errlist_t;

using namespace std;

/* Error code returned by sgx_create_enclave */
static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
    {
        SGX_ERROR_INVALID_PARAMETER,
        "Invalid parameter.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_MEMORY,
        "Out of memory.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_LOST,
        "Power transition occurred.",
        "Please refer to the sample \"PowerTransition\" for details."
    },
    {
        SGX_ERROR_INVALID_ENCLAVE,
        "Invalid enclave image.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ENCLAVE_ID,
        "Invalid enclave identification.",
        NULL
    },
    {
        SGX_ERROR_INVALID_SIGNATURE,
        "Invalid enclave signature.",
        NULL
    },
    {
        SGX_ERROR_OUT_OF_EPC,
        "Out of EPC memory.",
        NULL
    },
    {
        SGX_ERROR_NO_DEVICE,
        "Invalid SGX device.",
        "Please make sure SGX module is enabled in the BIOS, and install SGX driver afterwards."
    },
    {
        SGX_ERROR_MEMORY_MAP_CONFLICT,
        "Memory map conflicted.",
        NULL
    },
    {
        SGX_ERROR_INVALID_METADATA,
        "Invalid enclave metadata.",
        NULL
    },
    {
        SGX_ERROR_DEVICE_BUSY,
        "SGX device was busy.",
        NULL
    },
    {
        SGX_ERROR_INVALID_VERSION,
        "Enclave version was invalid.",
        NULL
    },
    {
        SGX_ERROR_INVALID_ATTRIBUTE,
        "Enclave was not authorized.",
        NULL
    },
    {
        SGX_ERROR_ENCLAVE_FILE_ACCESS,
        "Can't open enclave file.",
        NULL
    },
};

/* Check error conditions for loading enclave */
void print_error_message(sgx_status_t ret)
{
    size_t idx = 0;
    size_t ttl = sizeof sgx_errlist/sizeof sgx_errlist[0];

    for (idx = 0; idx < ttl; idx++) {
        if(ret == sgx_errlist[idx].err) {
            if(NULL != sgx_errlist[idx].sug)
                printf("Info: %s\n", sgx_errlist[idx].sug);
            printf("Error: %s\n", sgx_errlist[idx].msg);
            break;
        }
    }

    if (idx == ttl)
    	printf("Error code is 0x%X. Please refer to the \"Intel SGX SDK Developer Reference\" for more details.\n", ret);
}

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}

class InferenceInEnclaveRoutine : public Routine {
public:
    InferenceInEnclaveRoutine(sgx_enclave_id_t eid) : eid_(eid) {}
    int execute() override {
        int retVal;
        sgx_status_t ret = cnn_inference_f32_cpp(eid_, &retVal);
        if(ret != SGX_SUCCESS)
        {
            print_error_message(ret);
            return -1;
        }
        return 0;
    }

    std::string getName() override {
        return "Inference in enclave";
    }

private:
    sgx_enclave_id_t eid_;
};

class TrainingInEnclaveRoutine : public Routine {
public:
    TrainingInEnclaveRoutine(sgx_enclave_id_t eid) : eid_(eid) {}
    int execute() override {
        int retVal;
        sgx_status_t ret = cnn_training_f32_cpp(eid_, &retVal);
        if(ret != SGX_SUCCESS)
        {
            print_error_message(ret);
            return -1;
        }
        return 0;
    }

    std::string getName() override {
        return "Training in enclave";
    }

private:
    sgx_enclave_id_t eid_;
};

int main(int, char**) {
    benchmark(new InferenceRoutine());
    benchmark(new TrainingRoutine());

    sgx_status_t ret = SGX_SUCCESS;
    sgx_enclave_id_t eid = 0;
    ret = sgx_create_enclave(ENCLAVE_NAME, SGX_DEBUG_FLAG, NULL, NULL, &eid, NULL);
    if(ret != SGX_SUCCESS)
    {
        print_error_message(ret);
        return -1;
    }

    {
        // First invocation always takes a lot longer
        int retVal = 0;
        ret = cnn_training_f32_cpp(eid, &retVal);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            return -1;
        }
    }

    benchmark(new InferenceInEnclaveRoutine(eid));
    benchmark(new TrainingInEnclaveRoutine(eid));

    sgx_destroy_enclave(eid);

    cout << "Enter a character before exit ..." << endl;
    getchar();
    return 0;
}

