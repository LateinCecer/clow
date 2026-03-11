
struct char_vec {
public:
    __device__ char_vec(const unsigned char * __restrict__ input) {
        *reinterpret_cast<int2*>(data) = *reinterpret_cast<const int2*>(input);
    }
    __device__ void store(unsigned char * output) const {
        *reinterpret_cast<int2*>(output) = *reinterpret_cast<const int2*>(data);
    }

public:
    static constexpr int size = 8;
    unsigned char data[size];
};

__device__ const int test_func(int x) {
    return max(x, 1);
}
