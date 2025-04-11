#pragma once
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

/* tips: 1. use __syncthreads(), if only one thread allcoate the inital val in shared,; otherwise the other will read the wrong.
tips 2. size_t problem to compare with negtive number
*/

__constant__ int lm2idx_gpu[256];
//__constant__ int Nc_plm;
__constant__ int cu_OFFSET_EXPPHI;
__constant__ int cu_OFFSET_POWRIJ;
__constant__ int cu_NP;
__constant__ int cu_N_HARM;
__constant__ int cu_COLS;
__constant__ int cu_NC;
__constant__ size_t cu_DIM;




__device__ cuDoubleComplex operator*(const double& scalar, const cuDoubleComplex& complex) {
    return make_cuDoubleComplex(scalar * complex.x, scalar * complex.y);
}

__device__ cuComplex operator*(const float& scalar, const cuComplex& complex) {
    return make_cuComplex(scalar * complex.x, scalar * complex.y);
}

template <typename T>
cudaTextureObject_t createTexture2D(T* data, size_t pitch, int rows, int cols) {
    cudaChannelFormatDesc channelDesc;
    int channel = sizeof(T) / 4;
    if (channel == 1)
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    else if (channel == 2)
        channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    else if (channel == 3)
        channelDesc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
    else
        channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = data;
    resDesc.res.pitch2D.desc = channelDesc;
    resDesc.res.pitch2D.width = cols;
    resDesc.res.pitch2D.height = rows;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc texDesc = {};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    return texObj;
}

__inline__ __device__ double legendre(int l, int m, float x, cudaTextureObject_t plm) {
    if (l == 0) return 1.0;
    if (m > l) return 0.0;
    int idx = m + l * (l + 1) / 2;
    return (double)tex2D<float>(plm, x, idx);;
}

__inline__ __device__ double warpReduceSum(double val, int width) {
#pragma unroll
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, width);
    }
    return val;
}

__global__ void _matmult_cuda(const cuDoubleComplex* x, cuDoubleComplex* y, cudaTextureObject_t YS_tex1, cudaTextureObject_t YS_tex2,
    cudaTextureObject_t legendre_tex, cudaTextureObject_t powrij_tex, cudaTextureObject_t expphi_tex, const double* rp_in, const float3* xyz) {
    __shared__ cuDoubleComplex tmp[6];
    __shared__ double3 pow_rp;
    __shared__ int lp;
    __shared__ int mp;
    __shared__ int n_harm;
    __shared__ int Np;
    __shared__ int i;
    __shared__ int idx;
    __shared__ unsigned int iconta;
    __shared__ int offset_powrij;
    __shared__ int offset_expphi;
    int tid = threadIdx.x;
    //int l = 0;
    //int m = 0;
    //int i = blockIdx.x;
    //int n_harm = gridDim.y;
    //int Np = gridDim.x;
    cuDoubleComplex res[3];
    res[0] = make_cuDoubleComplex(0, 0);
    res[1] = res[0];
    res[2] = res[0];
    //float4 YS1 = make_float4(0, 0, 0, 0);
    //float4 YS2 = YS1;
    
    if (tid == 0) {
        i = blockIdx.x;
        idx = blockIdx.y;
        lp = lm2idx_gpu[2 * idx];
        mp = lm2idx_gpu[2 * idx + 1];
        n_harm = gridDim.y;
        Np = gridDim.x;
        offset_powrij = cu_OFFSET_POWRIJ;
        offset_expphi = cu_OFFSET_EXPPHI;
        double rp = rp_in[i];
        pow_rp = make_double3(pow(rp, lp - 1), pow(rp, lp), pow(rp, lp + 1));
        iconta = 3 * blockIdx.y + 3 * i * n_harm;
        res[0] = lp * (lp + 1) / pow_rp.y * x[iconta + 2];
        res[1] = cuCadd(lp / pow(rp, lp + 2) * x[iconta], -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y * x[iconta + 2]);
        res[2] = 1.0 / pow_rp.z * x[iconta + 1];
        for (int tmp1 = 0; tmp1 < 6; tmp1++) {
            tmp[tmp1] = make_cuDoubleComplex(0, 0);
        }
    }
    __syncthreads();
    //if (tid < n_harm) {
    int l = lm2idx_gpu[2 * tid] + lp;
    int m = lm2idx_gpu[2 * tid + 1] - mp;
    float4 YS1 = tex2D<float4>(YS_tex1, tid, idx);
    float4 YS2 = tex2D<float4>(YS_tex2, tid, idx);
    //}
    for (int j = 0; j < Np; j++) {
        if (i != j) {
            float3 xyz_index = xyz[i * Np + j];
            unsigned int jconta = 3 * (j * n_harm + tid);
            double plm0 = legendre(l - 2, abs(m), xyz_index.y, legendre_tex);
            double plm1 = legendre(l - 1, abs(m), xyz_index.y, legendre_tex);
            double plm2 = legendre(l, abs(m), xyz_index.y, legendre_tex);
            double rij0 = tex2D<float>(powrij_tex, xyz_index.x, -l - 1 + offset_powrij);
            double rij1 = tex2D<float>(powrij_tex, xyz_index.x, -l + offset_powrij);
            double rij2 = tex2D<float>(powrij_tex, xyz_index.x, -l + 1 + offset_powrij);
            float2 phi = tex2D<float2>(expphi_tex, xyz_index.z, m + offset_expphi);
            cuDoubleComplex exp_phi = make_cuDoubleComplex(phi.x, phi.y);
            cuDoubleComplex tmp0 = x[jconta];
            cuDoubleComplex tmp1 = x[jconta + 1];
            cuDoubleComplex tmp2 = x[jconta + 2];
            tmp0 = cuCmul(tmp0, exp_phi);
            tmp1 = cuCmul(tmp1, exp_phi);
            tmp2 = cuCmul(tmp2, exp_phi);
            double coeff = YS2.z * pow_rp.x * plm2 * rij0;
            res[0] = cuCadd(res[0], coeff * tmp0);
            coeff = YS1.w * pow_rp.x * plm1 * rij1;
            res[0] = cuCadd(res[0], coeff * make_cuDoubleComplex(-tmp1.y, tmp1.x));
            coeff = (YS1.x * pow_rp.x * plm2 + YS2.w * pow_rp.x * plm0) * rij2 + YS1.z * pow_rp.x * rij0 * plm2;
            res[0] = cuCadd(res[0], coeff * tmp2);
            coeff = YS2.y * pow_rp.z * rij0 * plm2;
            res[1] = cuCadd(res[1], coeff * tmp2);
            coeff = YS2.x * pow_rp.y * plm2 * rij0;
            res[2] = cuCadd(res[2], coeff * tmp1);
            coeff = YS1.y * pow_rp.y * plm1 * rij1;
            res[2] = cuCadd(res[2], coeff * make_cuDoubleComplex(-tmp2.y, tmp2.x));
        }
    }
    int size_warp = (tid >= 32) ? n_harm - 31 : 32;
    // sum reduced
    res[0].x = warpReduceSum(res[0].x, size_warp);
    res[0].y = warpReduceSum(res[0].y, size_warp);
    res[1].x = warpReduceSum(res[1].x, size_warp);
    res[1].y = warpReduceSum(res[1].y, size_warp);
    res[2].x = warpReduceSum(res[2].x, size_warp);
    res[2].y = warpReduceSum(res[2].y, size_warp);
    if (tid % warpSize == 0) {
        tmp[tid / warpSize * 3] = res[0];
        tmp[tid / warpSize * 3 + 1] = res[1];
        tmp[tid / warpSize * 3 + 2] = res[2];
    }
    //tmp[tid].x += __shfl_down_sync(0xffffffff, tmp[tid].x, offset, width);
    __syncthreads();
    if (tid == 0) {
        y[iconta] = cuCadd(tmp[0], tmp[3]);
        y[iconta + 1] = cuCadd(tmp[1], tmp[4]);
        y[iconta + 2] = cuCadd(tmp[2], tmp[5]);
    }
}

__global__ void _compute_matrix_cuda(cuDoubleComplex* Mc, cudaTextureObject_t YS_tex1, cudaTextureObject_t YS_tex2,
    cudaTextureObject_t legendre_tex, cudaTextureObject_t powrij_tex, cudaTextureObject_t expphi_tex, const double* rp_in, const float3* xyz) {
    __shared__ double3 pow_rp;
    __shared__ int lp;
    __shared__ int mp;
    __shared__ int n_harm;
    __shared__ int Np;
    __shared__ int iconta;
    __shared__ int offset_powrij;
    __shared__ int offset_expphi;
    __shared__ size_t idx;
    __shared__ size_t dim;
    int tid = threadIdx.x;
    int l = 1;
    int m = 0;
    int i = blockIdx.x;
    float4 YS1 = make_float4(0, 0, 0, 0);
    float4 YS2 = YS1;
    if (tid == 0) {
        n_harm = gridDim.y;
        Np = gridDim.x;
        //i = blockIdx.x;
        lp = lm2idx_gpu[2 * blockIdx.y];
        mp = lm2idx_gpu[2 * blockIdx.y + 1];
        double rp = rp_in[i];
        pow_rp = make_double3(pow(rp, lp - 1), pow(rp, lp), pow(rp, lp + 1));
        offset_powrij = cu_OFFSET_POWRIJ;
        offset_expphi = cu_OFFSET_EXPPHI;
        //pre_cols_i = cols * iconta;
        iconta = 3 * blockIdx.y + 3 * i * n_harm;// +icol * cu_DIM;
        //printf("iconta: %d", cu_DIM);
        idx = iconta * cu_DIM;
        dim = cu_DIM;
        double coeff = lp * (lp + 1.0) / pow_rp.y;
        Mc[idx + iconta + 2] = make_cuDoubleComplex(coeff, 0.0);
        coeff = lp / pow(rp, lp + 2);
        Mc[idx + dim + iconta] = make_cuDoubleComplex(coeff, 0.0);
        coeff = -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y;
        Mc[idx + dim + iconta + 2] = make_cuDoubleComplex(coeff, 0.0);
        coeff = 1.0 / pow_rp.z;
        Mc[idx + dim + dim + iconta + 1] = make_cuDoubleComplex(coeff, 0.0);
        
    }
    __syncthreads();
    if (tid < n_harm) {
        l = lm2idx_gpu[2 * tid];
        m = lm2idx_gpu[2 * tid + 1];
        int idy = m + l + l * l - 1;
        int idx = mp + lp + lp * lp - 1;
        YS1 = tex2D<float4>(YS_tex1, idy, idx);
        YS2 = tex2D<float4>(YS_tex2, idy, idx);
    }
    for (int j = 0; j < Np; j++) {
        if (tid < n_harm && i != j) {
            float3 xyz_index = xyz[i * Np + j];
            int jconta = 3 * (j * n_harm + tid);// +icol *cu_DIM;
            double plm0 = legendre(l + lp - 2, abs(m - mp), xyz_index.y, legendre_tex);
            double plm1 = legendre(l + lp - 1, abs(m - mp), xyz_index.y, legendre_tex);
            double plm2 = legendre(l + lp, abs(m - mp), xyz_index.y, legendre_tex);
            double rij0 = tex2D<float>(powrij_tex, xyz_index.x, -l - lp - 1 + offset_powrij);
            double rij1 = tex2D<float>(powrij_tex, xyz_index.x, -l - lp + offset_powrij);
            double rij2 = tex2D<float>(powrij_tex, xyz_index.x, -l - lp + 1 + offset_powrij);
            float2 phi = tex2D<float2>(expphi_tex, xyz_index.z, m - mp + offset_expphi);
            cuDoubleComplex exp_phi = make_cuDoubleComplex(phi.x, phi.y);
            cuDoubleComplex exp_phi2 = make_cuDoubleComplex(-exp_phi.y, exp_phi.x);
            double coeff = YS2.z * pow_rp.x * plm2 * rij0;
            Mc[idx + jconta] = coeff * exp_phi;
            
            coeff = YS1.w * pow_rp.x * plm1 * rij1;
            Mc[idx + jconta + 1] = coeff * exp_phi2;
            
            coeff = (YS1.x * pow_rp.x * plm2 + YS2.w * pow_rp.x * plm0) * rij2 + YS1.z * pow_rp.x * rij0 * plm2;
            Mc[idx + jconta + 2] = coeff * exp_phi;

            coeff = YS2.y * pow_rp.z * rij0 * plm2;
            Mc[idx + dim + jconta + 2] = coeff * exp_phi;
            
            coeff = YS2.x * pow_rp.y * plm2 * rij0;
            Mc[idx + dim + dim + jconta + 1] = coeff * exp_phi;
            
            coeff = YS1.y * pow_rp.y * plm1 * rij1;
            Mc[idx + dim + dim + jconta + 2] = coeff * exp_phi2;

        }
    }
}

__global__ void addComplexKernel(cuDoubleComplex* y, cuDoubleComplex c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = cuCadd(y[idx], c); 
    }
}

__global__ void _matmult2_cuda(const cuDoubleComplex* x, cuDoubleComplex* y, cudaTextureObject_t YS_tex1, cudaTextureObject_t YS_tex2,
    cudaTextureObject_t legendre_tex, cudaTextureObject_t powrij_tex, cudaTextureObject_t expphi_tex, const double* rp_in, const float3* xyz) {
    __shared__ double3 pow_rp;
    __shared__ int lp;
    __shared__ int mp;
    __shared__ int n_harm;
    __shared__ int Np;
    __shared__ int i;
    __shared__ int idx;
    __shared__ unsigned int iconta;
    __shared__ int offset_powrij;
    __shared__ int offset_expphi;
    int tid = threadIdx.x;
    cuDoubleComplex res[3];
    res[0] = make_cuDoubleComplex(0, 0);
    res[1] = res[0];
    res[2] = res[0];
    if (tid == 0) {
        i = blockIdx.x;
        idx = blockIdx.y;
        lp = lm2idx_gpu[2 * idx];
        mp = lm2idx_gpu[2 * idx + 1];
        n_harm = gridDim.y;
        Np = gridDim.x;
        offset_powrij = cu_OFFSET_POWRIJ;
        offset_expphi = cu_OFFSET_EXPPHI;
        double rp = rp_in[i];
        pow_rp = make_double3(pow(rp, lp - 1), pow(rp, lp), pow(rp, lp + 1));
        iconta = 3 * blockIdx.y + 3 * i * n_harm;
        res[0] = lp * (lp + 1) / pow_rp.y * x[iconta + 2];
        res[1] = cuCadd(lp / pow(rp, lp + 2) * x[iconta], -0.5 * lp * (lp + 1) * (2 * lp - 1) / pow_rp.y * x[iconta + 2]);
        res[2] = 1.0 / pow_rp.z * x[iconta + 1];
    }
    __syncthreads();
    int l = lm2idx_gpu[2 * tid] + lp;
    int m = lm2idx_gpu[2 * tid + 1] - mp;
    float4 YS1 = tex2D<float4>(YS_tex1, tid, idx);
    float4 YS2 = tex2D<float4>(YS_tex2, tid, idx);
    for (int j = 0; j < Np; j++) {
        if (i != j) {
            float3 xyz_index = xyz[i * Np + j];
            unsigned int jconta = 3 * (j * n_harm + tid);
            double plm0 = legendre(l - 2, abs(m), xyz_index.y, legendre_tex);
            double plm1 = legendre(l - 1, abs(m), xyz_index.y, legendre_tex);
            double plm2 = legendre(l, abs(m), xyz_index.y, legendre_tex);
            double rij0 = tex2D<float>(powrij_tex, xyz_index.x, -l - 1 + offset_powrij);
            double rij1 = tex2D<float>(powrij_tex, xyz_index.x, -l + offset_powrij);
            double rij2 = tex2D<float>(powrij_tex, xyz_index.x, -l + 1 + offset_powrij);
            float2 phi = tex2D<float2>(expphi_tex, xyz_index.z, m + offset_expphi);
            cuDoubleComplex exp_phi = make_cuDoubleComplex(phi.x, phi.y);
            cuDoubleComplex tmp0 = x[jconta];
            cuDoubleComplex tmp1 = x[jconta + 1];
            cuDoubleComplex tmp2 = x[jconta + 2];
            tmp0 = cuCmul(tmp0, exp_phi);
            tmp1 = cuCmul(tmp1, exp_phi);
            tmp2 = cuCmul(tmp2, exp_phi);
            double coeff = YS2.z * pow_rp.x * plm2 * rij0;
            res[0] = cuCadd(res[0], coeff * tmp0);
            coeff = YS1.w * pow_rp.x * plm1 * rij1;
            res[0] = cuCadd(res[0], coeff * make_cuDoubleComplex(-tmp1.y, tmp1.x));
            coeff = (YS1.x * pow_rp.x * plm2 + YS2.w * pow_rp.x * plm0) * rij2 + YS1.z * pow_rp.x * rij0 * plm2;
            res[0] = cuCadd(res[0], coeff * tmp2);
            coeff = YS2.y * pow_rp.z * rij0 * plm2;
            res[1] = cuCadd(res[1], coeff * tmp2);
            coeff = YS2.x * pow_rp.y * plm2 * rij0;
            res[2] = cuCadd(res[2], coeff * tmp1);
            coeff = YS1.y * pow_rp.y * plm1 * rij1;
            res[2] = cuCadd(res[2], coeff * make_cuDoubleComplex(-tmp2.y, tmp2.x));
        }
    }
    int size_warp = blockDim.x;
    // sum reduced
    res[0].x = warpReduceSum(res[0].x, size_warp);
    res[0].y = warpReduceSum(res[0].y, size_warp);
    res[1].x = warpReduceSum(res[1].x, size_warp);
    res[1].y = warpReduceSum(res[1].y, size_warp);
    res[2].x = warpReduceSum(res[2].x, size_warp);
    res[2].y = warpReduceSum(res[2].y, size_warp);
    if (tid == 0) {
        y[iconta] = res[0];
        y[iconta + 1] = res[1];
        y[iconta + 2] = res[2];
    }
}


__global__ void upAllocate(cuDoubleComplex* y, int yn_harm, int dim, cuDoubleComplex* x, int xn_harm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        int i = (idx / 3) / yn_harm;
        int lm = (idx / 3) % yn_harm;
        int rem = idx % 3;
        x[3 * (i * xn_harm + lm) + rem] = y[idx];
    }
}

__global__ void downAllocate(cuDoubleComplex* yb, int n_harm, int dim, cuDoubleComplex* x, int xn_harm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        int i = (idx / 3) / n_harm;
        int lm = (idx / 3) % n_harm;
        int rem = idx % 3;
        if(lm < xn_harm)
            x[3 * (i * xn_harm + lm) + rem] = yb[idx];
    }
}
