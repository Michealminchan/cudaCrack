#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

#define MD5_GET_UINT4(x) (			\
	(x < (N + 1) /2)?				\
	buf[x] : (						\
	(! (N % 2) && (x == N / 2))?	\
	(1U << 31U) : (					\
	(x == 14U) ?					\
	N * 16 : 0 )) )

template<size_t N>
__forceinline__ __device__ uint4 cudaMD5Op(const uint32_t(&buf)[(N + 1) / 2])
{
	const uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };

	uint32_t a = state[0], b = state[1], c = state[2], d = state[3];

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

	uint4 ret = { a, b, c, d };
	return ret;
}

template<size_t N>
__inline__ __device__ void initIdxString(const uint16_t (&base)[N], uint16_t (&idx)[N], uint32_t tid)
{
	uint32_t carry = tid;
#pragma unroll
	for (size_t i = 0; i < N; ++i)
	{
		carry += base[i];
		idx[i] = carry % CHARSET_SIZE;
		carry /= CHARSET_SIZE;
	}
}

template<size_t N>
__inline__ __device__ void nextIdxString(uint16_t(&idx)[N], uint32_t len)
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
__inline__ __device__ void cvtToRealStr(const uint16_t(&idx)[N], uint32_t(&str)[(N + 1) / 2])
{
#pragma unroll
	for (size_t i = 0; i < N / 2; ++i)
	{
		str[i] = idx[i * 2] + CHARSET_BEGIN + (idx[i * 2 + 1] + CHARSET_BEGIN << 16U);
	}
	if (N % 2)
	{
		str[(N - 1) / 2] = idx[N - 1] + CHARSET_BEGIN + (1U << 15U);
	}
}

template<size_t N>
__global__ void cudaCheckMD5(size_t times, const uint16_t(*base)[N])
{
	uint32_t ttotal = gridDim.x * blockDim.x;
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	//init
	uint16_t idx[N];
	initIdxString<N>(*base, idx, tid);
	for (size_t i = 0; i < times; ++i)
	{
		uint32_t str[(N + 1) / 2];
		cvtToRealStr<N>(idx, str);
		uint4 md5 = cudaMD5Op<N>(str);
		nextIdxString<N>(idx, ttotal);
	}
}

int main()
try
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		throw cudaStatus;
	}

	uint16_t pw[8] = {
		'q' - ' ', 
		'5' - ' ',
		'u' - ' ',
		'9' - ' ',
		'm' - ' ',
		'2' - ' ',
		'j' - ' ',
		'6' - ' '};
	decltype(&pw) _d_Passwd = nullptr;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&_d_Passwd, sizeof(*_d_Passwd));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		throw cudaStatus;
	}

	std::unique_ptr<decltype(pw), decltype(&cudaFree)>
		d_passwd(_d_Passwd, cudaFree);

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(_d_Passwd, pw, sizeof(*_d_Passwd), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		throw cudaStatus;
	}
	// Launch a kernel on the GPU with one thread for each element.
	cudaCheckMD5<_countof(pw)> << < 1, 256 >> > (0x1000, _d_Passwd);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		throw cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		throw cudaStatus;
	}
}
catch (cudaError_t e)
{
	return e;
}