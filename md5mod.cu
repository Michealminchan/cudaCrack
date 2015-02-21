#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename Fty>
struct nvapi_traits;

template<typename ...Args>
struct nvapi_traits<cudaError_t CUDARTAPI (Args...)>{
	typedef cudaError_t CUDARTAPI Fty(Args...);
	
	template<Fty F>
	static __forceinline void Function(Args... args)
	{
		cudaError_t ret = F(args...);
		if (ret != cudaSuccess)
		{
			throw ret;
		}
	}
};

#define GEN_NVAPI_EH(f, nf)\
	const auto& nf = nvapi_traits<decltype(f)>::Function<f>

__constant__ uint32_t d_target_md5[4];
__device__ uint64_t d_find = -1ULL;

/* Constants for MD5Transform routine.
*/

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

/* F, G, H and I are basic MD5 functions.
*/
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits.
*/
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
Rotation is separate from addition to prevent recomputation.
*/
#define FF(a, b, c, d, x, s, ac) { \
 (a) += F ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
    }
#define GG(a, b, c, d, x, s, ac) { \
 (a) += G ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
    }
#define HH(a, b, c, d, x, s, ac) { \
 (a) += H ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
    }
#define II(a, b, c, d, x, s, ac) { \
 (a) += I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
 (a) = ROTATE_LEFT ((a), (s)); \
 (a) += (b); \
    }

#define CHARSET_SIZE 95U
#define CHARSET_BEGIN 32U

/*static inline size_t next_idx(size_t& idx)
{
	size_t ret = idx % CHARSET_SIZE + CHARSET_BEGIN;
	idx /= CHARSET_SIZE;
	return ret;
}*/

#define MD5_GET_UINT4(x) (					\
	(x < N /2)?								\
		buf[x] : (							\
		(x == N / 2)? (						\
			(N % 2)?						\
				buf[N / 2] | (1U << 23U) :	\
				(1U << 7U)					\
			) :	(							\
			(x == 14U) ?					\
				N * 16 : 0					\
			)								\
		)									\
	)


namespace CUDA_CRACK{

	template<size_t N>
	__forceinline__ __device__ void cudaMD5Op(uint32_t(&out)[4], const uint32_t(&buf)[(N + 1) / 2])
	{
		const uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };

		uint32_t &a = out[0], &b = out[1], &c = out[2], &d = out[3];
		a = state[0], b = state[1], c = state[2], d = state[3];

		/* Round 1 */
		FF(a, b, c, d, MD5_GET_UINT4(0), S11, 0xd76aa478); /* 1 */
		FF(d, a, b, c, MD5_GET_UINT4(1), S12, 0xe8c7b756); /* 2 */
		FF(c, d, a, b, MD5_GET_UINT4(2), S13, 0x242070db); /* 3 */
		FF(b, c, d, a, MD5_GET_UINT4(3), S14, 0xc1bdceee); /* 4 */
		FF(a, b, c, d, MD5_GET_UINT4(4), S11, 0xf57c0faf); /* 5 */
		FF(d, a, b, c, MD5_GET_UINT4(5), S12, 0x4787c62a); /* 6 */
		FF(c, d, a, b, MD5_GET_UINT4(6), S13, 0xa8304613); /* 7 */
		FF(b, c, d, a, MD5_GET_UINT4(7), S14, 0xfd469501); /* 8 */
		FF(a, b, c, d, MD5_GET_UINT4(8), S11, 0x698098d8); /* 9 */
		FF(d, a, b, c, MD5_GET_UINT4(9), S12, 0x8b44f7af); /* 10 */
		FF(c, d, a, b, MD5_GET_UINT4(10), S13, 0xffff5bb1); /* 11 */
		FF(b, c, d, a, MD5_GET_UINT4(11), S14, 0x895cd7be); /* 12 */
		FF(a, b, c, d, MD5_GET_UINT4(12), S11, 0x6b901122); /* 13 */
		FF(d, a, b, c, MD5_GET_UINT4(13), S12, 0xfd987193); /* 14 */
		FF(c, d, a, b, MD5_GET_UINT4(14), S13, 0xa679438e); /* 15 */
		FF(b, c, d, a, MD5_GET_UINT4(15), S14, 0x49b40821); /* 16 */

		/* Round 2 */
		GG(a, b, c, d, MD5_GET_UINT4(1), S21, 0xf61e2562); /* 17 */
		GG(d, a, b, c, MD5_GET_UINT4(6), S22, 0xc040b340); /* 18 */
		GG(c, d, a, b, MD5_GET_UINT4(11), S23, 0x265e5a51); /* 19 */
		GG(b, c, d, a, MD5_GET_UINT4(0), S24, 0xe9b6c7aa); /* 20 */
		GG(a, b, c, d, MD5_GET_UINT4(5), S21, 0xd62f105d); /* 21 */
		GG(d, a, b, c, MD5_GET_UINT4(10), S22, 0x2441453); /* 22 */
		GG(c, d, a, b, MD5_GET_UINT4(15), S23, 0xd8a1e681); /* 23 */
		GG(b, c, d, a, MD5_GET_UINT4(4), S24, 0xe7d3fbc8); /* 24 */
		GG(a, b, c, d, MD5_GET_UINT4(9), S21, 0x21e1cde6); /* 25 */
		GG(d, a, b, c, MD5_GET_UINT4(14), S22, 0xc33707d6); /* 26 */
		GG(c, d, a, b, MD5_GET_UINT4(3), S23, 0xf4d50d87); /* 27 */
		GG(b, c, d, a, MD5_GET_UINT4(8), S24, 0x455a14ed); /* 28 */
		GG(a, b, c, d, MD5_GET_UINT4(13), S21, 0xa9e3e905); /* 29 */
		GG(d, a, b, c, MD5_GET_UINT4(2), S22, 0xfcefa3f8); /* 30 */
		GG(c, d, a, b, MD5_GET_UINT4(7), S23, 0x676f02d9); /* 31 */
		GG(b, c, d, a, MD5_GET_UINT4(12), S24, 0x8d2a4c8a); /* 32 */

		/* Round 3 */
		HH(a, b, c, d, MD5_GET_UINT4(5), S31, 0xfffa3942); /* 33 */
		HH(d, a, b, c, MD5_GET_UINT4(8), S32, 0x8771f681); /* 34 */
		HH(c, d, a, b, MD5_GET_UINT4(11), S33, 0x6d9d6122); /* 35 */
		HH(b, c, d, a, MD5_GET_UINT4(14), S34, 0xfde5380c); /* 36 */
		HH(a, b, c, d, MD5_GET_UINT4(1), S31, 0xa4beea44); /* 37 */
		HH(d, a, b, c, MD5_GET_UINT4(4), S32, 0x4bdecfa9); /* 38 */
		HH(c, d, a, b, MD5_GET_UINT4(7), S33, 0xf6bb4b60); /* 39 */
		HH(b, c, d, a, MD5_GET_UINT4(10), S34, 0xbebfbc70); /* 40 */
		HH(a, b, c, d, MD5_GET_UINT4(13), S31, 0x289b7ec6); /* 41 */
		HH(d, a, b, c, MD5_GET_UINT4(0), S32, 0xeaa127fa); /* 42 */
		HH(c, d, a, b, MD5_GET_UINT4(3), S33, 0xd4ef3085); /* 43 */
		HH(b, c, d, a, MD5_GET_UINT4(6), S34, 0x4881d05); /* 44 */
		HH(a, b, c, d, MD5_GET_UINT4(9), S31, 0xd9d4d039); /* 45 */
		HH(d, a, b, c, MD5_GET_UINT4(12), S32, 0xe6db99e5); /* 46 */
		HH(c, d, a, b, MD5_GET_UINT4(15), S33, 0x1fa27cf8); /* 47 */
		HH(b, c, d, a, MD5_GET_UINT4(2), S34, 0xc4ac5665); /* 48 */

		/* Round 4 */
		II(a, b, c, d, MD5_GET_UINT4(0), S41, 0xf4292244); /* 49 */
		II(d, a, b, c, MD5_GET_UINT4(7), S42, 0x432aff97); /* 50 */
		II(c, d, a, b, MD5_GET_UINT4(14), S43, 0xab9423a7); /* 51 */
		II(b, c, d, a, MD5_GET_UINT4(5), S44, 0xfc93a039); /* 52 */
		II(a, b, c, d, MD5_GET_UINT4(12), S41, 0x655b59c3); /* 53 */
		II(d, a, b, c, MD5_GET_UINT4(3), S42, 0x8f0ccc92); /* 54 */
		II(c, d, a, b, MD5_GET_UINT4(10), S43, 0xffeff47d); /* 55 */
		II(b, c, d, a, MD5_GET_UINT4(1), S44, 0x85845dd1); /* 56 */
		II(a, b, c, d, MD5_GET_UINT4(8), S41, 0x6fa87e4f); /* 57 */
		II(d, a, b, c, MD5_GET_UINT4(15), S42, 0xfe2ce6e0); /* 58 */
		II(c, d, a, b, MD5_GET_UINT4(6), S43, 0xa3014314); /* 59 */
		II(b, c, d, a, MD5_GET_UINT4(13), S44, 0x4e0811a1); /* 60 */
		II(a, b, c, d, MD5_GET_UINT4(4), S41, 0xf7537e82); /* 61 */
		II(d, a, b, c, MD5_GET_UINT4(11), S42, 0xbd3af235); /* 62 */
		II(c, d, a, b, MD5_GET_UINT4(2), S43, 0x2ad7d2bb); /* 63 */
		II(b, c, d, a, MD5_GET_UINT4(9), S44, 0xeb86d391); /* 64 */

		a += state[0];
		b += state[1];
		c += state[2];
		d += state[3];

	}

	template<size_t N>
	__forceinline__ __device__ void cudaNextIdxString(uint16_t(&idx)[N], uint32_t len)
	{
		uint32_t carry = len;
#pragma unroll
		for (size_t i = 0; i < N; ++i)
		{
			carry += idx[i];
			idx[i] = carry % CHARSET_SIZE;
			carry /= CHARSET_SIZE;
		}
	}

	template<size_t N>
	__forceinline__ __device__ void cudaCvtToRealStr(const uint16_t(&idx)[N], uint32_t(&str)[(N + 1) / 2])
	{
#pragma unroll
		for (size_t i = 0; i < N / 2; ++i)
		{
			str[i] = idx[i * 2] + CHARSET_BEGIN + (idx[i * 2 + 1] + CHARSET_BEGIN << 16U);
		}
		if (N % 2)
		{
			str[(N - 1) / 2] = idx[N - 1] + CHARSET_BEGIN;
		}
	}

	template<size_t N>
	__forceinline__ __device__ bool cudaMemCmp(const uint32_t(&a)[N], const uint32_t(&b)[N])
	{
#pragma unroll
		for (size_t i = 0; i < N; ++i)
		{
			if (a[i] != b[i])
			{
				return true;
			}
		}
		return false;
	}

	template<size_t N>
	__global__ void cudaCheckMD5(uint64_t times, const uint16_t(&base)[N])
	{
		uint32_t ttotal = gridDim.x * blockDim.x;
		uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

		//init
		uint16_t idx[N];
#pragma unroll
		for (size_t i = 0; i < N; ++i)
		{
			idx[i] = base[i];
		}
		cudaNextIdxString<N>(idx, tid);
		for (uint64_t i = 0; i < times; ++i)
		{
			uint32_t str[(N + 1) / 2];
			cudaCvtToRealStr<N>(idx, str);
			uint32_t md5[4];
			cudaMD5Op<N>(md5, str);
			cudaNextIdxString<N>(idx, ttotal);
			if (!cudaMemCmp(md5, d_target_md5))
			{
				d_find = i* ttotal + tid;
			}
		}
	}
}

GEN_NVAPI_EH(cudaSetDevice, EcudaSetDevice);
GEN_NVAPI_EH(::cudaMalloc, EcudaMalloc);
GEN_NVAPI_EH(cudaMemcpy, EcudaMemcpy);
GEN_NVAPI_EH(cudaDeviceSynchronize, EcudaDeviceSynchronize);
GEN_NVAPI_EH(cudaGetLastError, EcudaCheckError);
GEN_NVAPI_EH(::cudaMemcpyToSymbol, EcudaMemcpyToSymbol);
GEN_NVAPI_EH(::cudaMemcpyFromSymbol, EcudaMemcpyFromSymbol);
GEN_NVAPI_EH(cudaGetDeviceProperties, EcudaGetDeviceProperties);

namespace{

	uint32_t gDim;
	uint32_t bDim;

	unsigned char htoi(char a) {
		if (a >= '0' && a <= '9') {
			return a - '0';
		}
		else if (a >= 'a' && a <= 'z') {
			return a - 'a' + 0xa;
		}
		else if (a >= 'A' && a <= 'Z') {
			return a - 'A' + 0xA;
		}
	}

	unsigned char h2toi(const char(&h2)[2])
	{
		return (htoi(h2[0]) << 4U) + htoi(h2[1]);
	}

	void ParseInputMD5(const char* str, uint32_t(&md5)[4])
	{
		typedef char MD5Arr[16][2];
		if (strlen(str) != 32){
			throw std::runtime_error("invalid md5 string");
		}
		const MD5Arr* in = (const MD5Arr*)str;
		unsigned char* buf = (unsigned char*)&md5;
		for (size_t i = 0; i < 16; ++i)
		{
			buf[i] = h2toi((*in)[i]);
		}
	}

	std::vector<uint16_t> ParseInputBase(const char* str)
	{
		typedef char BaseChar[2];
		size_t len = strlen(str);
		if (len % 2){
			throw std::runtime_error("invalid base string");
		}
		const BaseChar* in = (const BaseChar*)str;
		std::vector<uint16_t> ret(len / 2);
		for (size_t i = 0; i != len / 2; ++i)
		{
			ret[i] = h2toi(in[i]);
		}
		return std::move(ret);
	}

	void NextIdxString(size_t N, uint16_t* idx, uint64_t len)
	{
		uint64_t carry = len;
		for (size_t i = 0; i < N; ++i)
		{
			carry += idx[i];
			idx[i] = carry % CHARSET_SIZE;
			carry /= CHARSET_SIZE;
		}
	}

	template<size_t N>
	uint64_t CrackOnce(uint32_t dim_g, uint32_t dim_b,uint16_t* h_baseidx, uint16_t* d_baseidx, uint64_t round)
	{
		typedef uint16_t IdxArrTy[N];
		EcudaMemcpy(d_baseidx, h_baseidx, sizeof(IdxArrTy), cudaMemcpyHostToDevice);
		IdxArrTy* d_baseidxarr = (IdxArrTy*)d_baseidx;
		CUDA_CRACK::cudaCheckMD5<N><<<dim_g, dim_b>>>(round, *d_baseidxarr);
		EcudaCheckError();
		EcudaDeviceSynchronize();
		uint64_t ret;
		EcudaMemcpyFromSymbol(&ret, &d_find, sizeof(d_find), 0, cudaMemcpyDeviceToHost);
		return ret;
	}

	decltype(CrackOnce<1>)* const CrackFs[] =
	{
		nullptr, CrackOnce < 1 >, CrackOnce< 2 >, CrackOnce < 3 >,
		CrackOnce< 4 >, CrackOnce < 5 >, CrackOnce< 6 >, CrackOnce < 7 >,
		CrackOnce< 8 >, CrackOnce < 9 >, CrackOnce< 10 >, CrackOnce < 11 >,
		CrackOnce< 12 >, CrackOnce < 13 >, CrackOnce< 14 >, CrackOnce < 15 >,
		CrackOnce< 16 >, CrackOnce < 17 >, CrackOnce< 18 >, CrackOnce < 19 >,
	};

	uint64_t CrackMd5(size_t len, uint16_t* h_baseidx, uint16_t* d_baseidx, uint64_t round)
	{
		uint64_t checked_times = 0;
		uint32_t dim_hi = gDim * bDim;
		uint64_t times_hi = (round / dim_hi);
		uint64_t total_hi = round - round % dim_hi;

		uint32_t dim_mid = bDim;
		uint32_t times_mid = (round % dim_hi) / dim_mid;
		uint32_t total_mid = dim_mid * times_mid;

		uint32_t dim_lo = round % dim_hi % dim_mid;

		if (times_hi)
		{
			uint64_t ret = CrackFs[len](gDim, bDim, h_baseidx, d_baseidx, times_hi);
			if (ret != -1ULL)
			{
				return checked_times + ret;
			}
			checked_times += total_hi;
		}
		if (times_mid)
		{
			NextIdxString(len, h_baseidx, total_hi);
			uint64_t ret = CrackFs[len](1, bDim, h_baseidx, d_baseidx, times_mid);
			if (ret != -1ULL)
			{
				return checked_times + ret;
			}
			checked_times += total_mid;
		}
		if (dim_lo)
		{
			NextIdxString(len, h_baseidx, total_mid);
			uint64_t ret = CrackFs[len](1, dim_lo, h_baseidx, d_baseidx, 1);
			if (ret != -1ULL)
			{
				return checked_times +ret;
			}
		}
		return -1ULL;
	}
}

int main(int, char** argv)
try
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	EcudaSetDevice(0);

	cudaDeviceProp devprop;
	EcudaGetDeviceProperties(&devprop, 0);

	bDim = 256U;
	gDim = devprop.maxThreadsPerMultiProcessor;
	gDim /= bDim;
	gDim *= devprop.multiProcessorCount;

	uint32_t h_targetmd5[4];
	ParseInputMD5(argv[1], h_targetmd5);
	EcudaMemcpyToSymbol(d_target_md5, h_targetmd5, sizeof(h_targetmd5), 0, cudaMemcpyHostToDevice);

	std::vector<uint16_t> h_baseidx = ParseInputBase(argv[2]);
	uint64_t round = atoll(argv[3]);

	uint16_t* d_baseidx;

	EcudaMalloc((void**)&d_baseidx, sizeof(uint16_t)*h_baseidx.size());
	printf("malloc success!\n");

	std::unique_ptr<uint16_t, decltype(&cudaFree)> d_passwd(d_baseidx, cudaFree);

	uint64_t find = CrackMd5(h_baseidx.size(), &h_baseidx[0], d_baseidx, round);
	if (find != -1ULL)
	{
		printf("found @%llx\n", find);
		return 1;
	}
	else
	{
		printf("not found\n");
		return -1;
	}
}
catch (cudaError_t e)
{
	fprintf(stderr, "error occurred: %s\n", cudaGetErrorString(e));
	return 255;
}