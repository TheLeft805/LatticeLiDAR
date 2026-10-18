// Explicit p48 posterior profile. One half warp per ANS lane; the release fixes 32 lanes.
// No cross-lane barrier in the symbol loop. A failure poisons the frame and
// the host must reject it before committing the group's symbols.
#include <cuda_runtime.h>
#include <stdint.h>
#ifndef JOINT_LANES
#define JOINT_LANES 16
#endif
static int execution_ctas=1;
extern "C" void joint_set_ctas(int value) { execution_ctas=value; }

// All callers have a quotient <=65532. A floating estimate is followed by
// integer overflow/range corrections; the returned quotient and remainder are
// exact, irrespective of the estimate's rounding direction.
__device__ __forceinline__ uint32_t small_div(uint64_t n,uint64_t d,uint64_t& remainder) {
    uint32_t q=min(uint32_t(__fdividef(float(n),float(d))),65532u);
    while(__umul64hi(uint64_t(q),d)){--q;}
    uint64_t product=uint64_t(q)*d;
    while(product>n){--q;product-=d;}
    while(n-product>=d){++q;product+=d;}
    remainder=n-product;return q;
}

__device__ __forceinline__ bool renorm(uint32_t& state,int& pos,const uint8_t* data) {
    // A p<=16 normalized state needs at most two bytes. Both addresses depend
    // only on the current stack position, so their loads can overlap.
    uint32_t first=pos>0?data[pos-1]:0,second=pos>1?data[pos-2]:0;
    if(state<(1u<<23)){if(pos<=0)return false;state=(state<<8)|first;--pos;}
    if(state<(1u<<23)){if(pos<=0)return false;state=(state<<8)|second;--pos;}
    if(state<(1u<<23))return false;
    return true;
}

template<bool PACKED_CDF16 = false, bool DENSE_CDF16 = false>
__global__ void decode_group(const int* base,const int* comp,const int* priors,
    const int* lengths,const int* offsets,int blocks,int dynamic,int k4,int active,int carry,
    uint32_t* states,int* positions,uint8_t* storage,int capacity,int16_t* output,
    uint64_t* posteriors,int* error, int row_count) {
    if(*error)return;
    int tid=threadIdx.x,lane=blockIdx.x*(blockDim.x/16)+tid/16,cls=tid%16;
    unsigned mask=0xffffu<<(((tid/16)&1)*16);
    int width=(blocks+JOINT_LANES-1)/JOINT_LANES;
    uint32_t state=states[lane];int pos=positions[lane];
    uint8_t* data=storage+lane*capacity;
    uint64_t old[4]={0,0,0,0};
    for(int b=lane*width;b<min(blocks,(lane+1)*width);++b) {
        int prior[4]={0,0,0,0};uint64_t mass[4]={0,0,0,0};int z=-1;
        if(k4) {
            int total=0,best=0;uint64_t den=old[0]+old[1]+old[2]+old[3];
            #pragma unroll
            for(int k=0;k<4;++k) {
                int value=priors[b*4+k];
                if(carry && den) {
                    uint64_t a=31u*value, v=62*old[k],rem;
                    uint32_t quotient=small_div(v,den,rem);
                    value=int(a/64+quotient+((a%64)*den+64*rem>=64*den)+1);
                }
                prior[k]=value;total+=value;
                if(value>prior[best])best=k;
            }
            if(carry && den)prior[best]+=128-total;
            #pragma unroll
            for(int k=0;k<4;++k)mass[k]=prior[k];
            if(active) {
                int slot=state&127,start=0;
                #pragma unroll
                for(int k=0;k<4;++k) {
                    if(z<0 && slot<start+prior[k]){z=k;state=prior[k]*(state>>7)+slot-start;}
                    start+=prior[k];
                }
                if(z<0 || !renorm(state,pos,data)){if(cls==0)atomicOr(error,1);return;}
            }
        }
        int previous=0;
        const int length = PACKED_CDF16 ? min(32, row_count-b*32) : lengths[b];
        for(int j=0;j<length;++j) {
            const int bi=((b*32+j)*(dynamic?16:1)+(dynamic?previous:0))*17;
            const int ci=((b*32+j)*16+previous)*68;
            const int* br=base+bi;
            const int* cr=k4?comp+ci:nullptr;
            const uint16_t* br16=reinterpret_cast<const uint16_t*>(base)+bi;
            const uint16_t* cr16=k4?reinterpret_cast<const uint16_t*>(comp)+ci:nullptr;
            int lo=0,hi=0,component_frequency[4];
            if(k4) {
                // Preload all four class intervals before finding the symbol.
                // Their addresses are independent of this symbol's ANS state.
                #pragma unroll
                for(int k=0;k<4;++k) {
                    int l=DENSE_CDF16 ? cr16[k*17+cls] : cr[k*17+cls];
                    int h=DENSE_CDF16 ? (cls==15 ? 65536 : cr16[k*17+cls+1]) : cr[k*17+cls+1];
                    component_frequency[k]=h-l;
                    if(k==z){lo=l;hi=h;}
                }
            }
            if (PACKED_CDF16) {
                const uint16_t* q = reinterpret_cast<const uint16_t*>(base)+(b*32+j)*17;
                lo=q[cls]; hi=cls==15 ? 65536 : q[cls+1];
            } else if(!k4||!active){
                lo=DENSE_CDF16 ? br16[cls] : br[cls];
                hi=DENSE_CDF16 ? (cls==15 ? 65536 : br16[cls+1]) : br[cls+1];
            }
            uint32_t slot=state&65535;
            unsigned hit=__ballot_sync(mask,lo<=slot && slot<hi);
            int y=__ffs(hit>>(((tid/16)&1)*16))-1;
            if(y<0){if(cls==0)atomicOr(error,2);return;}
            int start=__shfl_sync(mask,lo,y,16),freq=__shfl_sync(mask,hi-lo,y,16);
            state=uint32_t(freq)*(state>>16)+slot-start;
            if(!renorm(state,pos,data)){if(cls==0)atomicOr(error,1);return;}
            if(cls==0)output[(PACKED_CDF16 ? b*32 : offsets[b])+j]=int16_t(y);
            if(k4) {
                uint64_t maximum=0;
                #pragma unroll
                for(int k=0;k<4;++k) {
                    int f=component_frequency[k];
                    mass[k]*=uint32_t(__shfl_sync(mask,f,y,16));
                    maximum=max(maximum,mass[k]);
                }
                int shift=max(0,64-__clzll(maximum)-48);
                #pragma unroll
                for(int k=0;k<4;++k)mass[k]=max(uint64_t(1),mass[k]>>shift);
            }
            previous=y;
        }
        if(k4) {
            uint64_t den=mass[0]+mass[1]+mass[2]+mass[3],remainder[4];
            int freq[4],total=0;
            #pragma unroll
            for(int k=0;k<4;++k) {
                old[k]=mass[k];
                if(cls==0)posteriors[b*4+k]=mass[k];
                uint64_t scaled=65532*mass[k];
                freq[k]=int(small_div(scaled,den,remainder[k]))+1;total+=freq[k];
            }
            int remain=65536-total;
            #pragma unroll
            for(int k=0;k<4;++k) {
                int rank=0;
                #pragma unroll
                for(int l=0;l<4;++l)rank+=(remainder[l]>remainder[k] || (remainder[l]==remainder[k] && l<k));
                freq[k]+=rank<remain;
            }
            if(active) {
                uint32_t start=0,f=freq[z];
                #pragma unroll
                for(int k=0;k<4;++k)if(k<z)start+=freq[k];
                uint32_t threshold=((1u<<23)>>16)*256*f;
                while(state>=threshold) {
                    if(pos>=capacity){if(cls==0)atomicOr(error,4);return;}
                    if(cls==0)data[pos]=uint8_t(state);
                    ++pos;state>>=8;
                }
                state=((state/f)<<16)+state%f+start;
                __syncwarp(mask);
            }
        }
    }
    if(cls==0){states[lane]=state;positions[lane]=pos;}
}

extern "C" int joint_decode(const void* base,const void* comp,const void* priors,
    const void* lengths,const void* offsets,int blocks,int dynamic,int k4,int active,int carry,
    void* states,void* positions,void* storage,int capacity,void* output,void* posterior,void* error,void* stream) {
    decode_group<false><<<execution_ctas,16*JOINT_LANES/execution_ctas,0,(cudaStream_t)stream>>>((const int*)base,(const int*)comp,(const int*)priors,
        (const int*)lengths,(const int*)offsets,blocks,dynamic,k4,active,carry,
        (uint32_t*)states,(int*)positions,(uint8_t*)storage,capacity,(int16_t*)output,
        (uint64_t*)posterior,(int*)error,0);
    return int(cudaGetLastError());
}

__global__ void prepare_groups(const int* intervals,const int* priors,const int* lengths,
    int blocks,int carry,int* paid,int* post) {
    int lane=blockIdx.x*(blockDim.x/16)+threadIdx.x/16,cls=threadIdx.x%16,width=(blocks+JOINT_LANES-1)/JOINT_LANES;
    uint64_t old[4]={0,0,0,0};
    for(int b=lane*width;b<min(blocks,(lane+1)*width);++b) {
        int prior[4],total=0,best=0;uint64_t mass[4],den=old[0]+old[1]+old[2]+old[3];
        #pragma unroll
        for(int k=0;k<4;++k) {
            int value=priors[b*4+k];
            if(carry&&den) {
                uint64_t a=31u*value,v=62*old[k],rem;
                uint32_t quotient=small_div(v,den,rem);
                value=int(a/64+quotient+((a%64)*den+64*rem>=64*den)+1);
            }
            prior[k]=value;total+=value;if(value>prior[best])best=k;
        }
        if(carry&&den)prior[best]+=128-total;
        #pragma unroll
        for(int k=0;k<4;++k){mass[k]=prior[k];if(cls==0)paid[b*4+k]=prior[k];}
        for(int j=0;j<lengths[b];++j) {
            uint64_t maximum=0;
            #pragma unroll
            for(int k=0;k<4;++k){mass[k]*=intervals[((b*32+j)*4+k)*2+1];maximum=max(maximum,mass[k]);}
            int shift=max(0,64-__clzll(maximum)-48);
            #pragma unroll
            for(int k=0;k<4;++k)mass[k]=max(uint64_t(1),mass[k]>>shift);
        }
        den=mass[0]+mass[1]+mass[2]+mass[3];uint64_t remainder[4];int freq[4];total=0;
        #pragma unroll
        for(int k=0;k<4;++k){old[k]=mass[k];uint64_t v=65532*mass[k];freq[k]=int(small_div(v,den,remainder[k]))+1;total+=freq[k];}
        int remaining=65536-total,cumulative=0;
        if(cls==0)post[b*5]=0;
        #pragma unroll
        for(int k=0;k<4;++k) {
            int rank=0;
            #pragma unroll
            for(int l=0;l<4;++l)rank+=(remainder[l]>remainder[k]||(remainder[l]==remainder[k]&&l<k));
            cumulative+=freq[k]+(rank<remaining);
            if(cls==0)post[b*5+k+1]=cumulative;
        }
    }
}

__device__ __forceinline__ bool put_interval(uint32_t& state,int& pos,uint8_t* data,
    int capacity,int cls,unsigned mask,uint32_t start,uint32_t freq,int precision) {
    uint32_t threshold=(((1u<<23)>>precision)<<8)*freq;
    while(state>=threshold) {
        if(pos>=capacity)return false;
        if(cls==0)data[pos]=uint8_t(state);++pos;state>>=8;
    }
    state=((state/freq)<<precision)+state%freq+start;
    __syncwarp(mask);return true;
}

__global__ void encode_group(const int* base,const int* comp,const int* lengths,
    const int* paid,const int* post,int blocks,int k4,int active,
    uint32_t* states,int* positions,uint8_t* storage,int capacity,int* error) {
    if(*error)return;
    int lane=blockIdx.x*(blockDim.x/16)+threadIdx.x/16,cls=threadIdx.x%16,width=(blocks+JOINT_LANES-1)/JOINT_LANES;
    unsigned mask=0xffffu<<(((threadIdx.x/16)&1)*16);
    uint32_t state=states[lane];int pos=positions[lane];uint8_t* data=storage+lane*capacity;
    for(int b=min(blocks,(lane+1)*width)-1;b>=lane*width;--b) {
        int z=-1;
        if(k4&&active) {
            uint32_t slot=state&65535;
            #pragma unroll
            for(int k=0;k<4;++k)if(z<0&&slot<uint32_t(post[b*5+k+1])) {
                z=k;state=uint32_t(post[b*5+k+1]-post[b*5+k])*(state>>16)+slot-post[b*5+k];
            }
            if(z<0||!renorm(state,pos,data)){if(cls==0)atomicOr(error,1);return;}
        }
        for(int j=lengths[b]-1;j>=0;--j) {
            const int* row=k4&&active?comp+((b*32+j)*4+z)*2:base+(b*32+j)*2;
            if(!put_interval(state,pos,data,capacity,cls,mask,row[0],row[1],16)){if(cls==0)atomicOr(error,4);return;}
        }
        if(k4&&active) {
            uint32_t start=0;
            #pragma unroll
            for(int k=0;k<4;++k)if(k<z)start+=paid[b*4+k];
            if(!put_interval(state,pos,data,capacity,cls,mask,start,paid[b*4+z],7)){if(cls==0)atomicOr(error,4);return;}
        }
    }
    if(cls==0){states[lane]=state;positions[lane]=pos;}
}

extern "C" int joint_prepare(const void* intervals,const void* priors,const void* lengths,
    int blocks,int carry,void* paid,void* post,void* stream) {
    prepare_groups<<<execution_ctas,16*JOINT_LANES/execution_ctas,0,(cudaStream_t)stream>>>((const int*)intervals,(const int*)priors,(const int*)lengths,
        blocks,carry,(int*)paid,(int*)post);
    return int(cudaGetLastError());
}

extern "C" int joint_encode(const void* base,const void* comp,const void* lengths,const void* paid,const void* post,
    int blocks,int k4,int active,void* states,void* positions,void* storage,int capacity,void* error,void* stream) {
    encode_group<<<execution_ctas,16*JOINT_LANES/execution_ctas,0,(cudaStream_t)stream>>>((const int*)base,(const int*)comp,(const int*)lengths,
        (const int*)paid,(const int*)post,blocks,k4,active,(uint32_t*)states,(int*)positions,(uint8_t*)storage,capacity,(int*)error);
    return int(cudaGetLastError());
}

// Host-side batching only. Existing kernels, stream ordering, and lane states
// are unchanged. Metadata stays on the host and is consumed before return.
extern "C" int joint_encode_many(const uint64_t* descriptors, int count, int prepare,
    int active, int carry, void* states, void* positions, void* storage,
    int capacity, void* error, void* stream) {
    if (prepare) {
        for (int i = 0; i < count; ++i) {
            const uint64_t* d = descriptors + i * 8;
            if (!d[7]) continue;
            int code = joint_prepare((const void*)d[1], (const void*)d[5],
                (const void*)d[2], int(d[6]), carry, (void*)d[3], (void*)d[4], stream);
            if (code) return code;
        }
    }
    for (int i = count - 1; i >= 0; --i) {
        const uint64_t* d = descriptors + i * 8;
        int code = joint_encode((const void*)d[0], (const void*)d[1],
            (const void*)d[2], (const void*)d[3], (const void*)d[4], int(d[6]), int(d[7]),
            active, states, positions, storage, capacity, error, stream);
        if (code) return code;
    }
    return 0;
}

extern "C" int joint_decode_cdf16(const void* cdf, int rows, void* states,
    void* positions, void* storage, int capacity, void* output, void* error, void* stream) {
    decode_group<true><<<execution_ctas,16*JOINT_LANES/execution_ctas,0,(cudaStream_t)stream>>>(
        (const int*)cdf,nullptr,nullptr,nullptr,nullptr,(rows+31)/32,0,0,0,0,
        (uint32_t*)states,(int*)positions,(uint8_t*)storage,capacity,(int16_t*)output,
        nullptr,(int*)error,rows);
    return int(cudaGetLastError());
}


extern "C" int joint_decode_dense_cdf16(const void* base,const void* comp,const void* priors,
    const void* lengths,const void* offsets,int blocks,int dynamic,int k4,int active,int carry,
    void* states,void* positions,void* storage,int capacity,void* output,void* posterior,void* error,void* stream) {
    decode_group<false, true><<<execution_ctas,16*JOINT_LANES/execution_ctas,0,(cudaStream_t)stream>>>((const int*)base,(const int*)comp,(const int*)priors,
        (const int*)lengths,(const int*)offsets,blocks,dynamic,k4,active,carry,
        (uint32_t*)states,(int*)positions,(uint8_t*)storage,capacity,(int16_t*)output,
        (uint64_t*)posterior,(int*)error,0);
    return int(cudaGetLastError());
}
