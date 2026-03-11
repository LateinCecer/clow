#include "testlib.cuh"

extern "C"
__global__ void gpu_div_kernel_vec(
    const unsigned char * __restrict__ input,
    unsigned char * __restrict__ output,
    int div
) {
    auto d = test_func(div);
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    char_vec vec(input + tid * 8);

    unsigned char *vec_data = vec.data;
    for (int i = 0; i < vec.size; i++) {
        vec_data[i] /= d;
    }

    vec.store(output + tid * 8);
}
