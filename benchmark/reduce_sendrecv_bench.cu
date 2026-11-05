#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define MPI_CHECK(x) do { int e=(x); if(e!=MPI_SUCCESS){fprintf(stderr,"MPI error %d\n",e); MPI_Abort(MPI_COMM_WORLD,e);} } while(0)
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define NCCL_CHECK(x) do { ncclResult_t e=(x); if(e!=ncclSuccess){fprintf(stderr,"NCCL: %s\n",ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

using Clock=std::chrono::steady_clock;
static double elapsedUs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double,std::micro>(b-a).count();
}
enum class Operation { AllReduce, SendRecv };

static void launch(Operation op, const float* sendbuf, float* recvbuf,
                   size_t count, int rank, ncclComm_t comm, cudaStream_t stream) {
  if (op == Operation::AllReduce) {
    NCCL_CHECK(ncclAllReduce(sendbuf,recvbuf,count,ncclFloat,ncclSum,comm,stream));
  } else {
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(sendbuf,count,ncclFloat,1-rank,comm,stream));
    NCCL_CHECK(ncclRecv(recvbuf,count,ncclFloat,1-rank,comm,stream));
    NCCL_CHECK(ncclGroupEnd());
  }
}

static bool validate(Operation op, const std::vector<float>& out, int rank) {
  const float expected = op==Operation::AllReduce ? 3.0f : float(2-rank);
  for(size_t i=0;i<out.size();++i) {
    if(std::fabs(out[i]-expected)>1e-5f) {
      fprintf(stderr,"rank %d mismatch at %zu: got %g expected %g\n",rank,i,out[i],expected);
      return false;
    }
  }
  return true;
}

static void runCase(Operation op,size_t bytes,int rank,ncclComm_t comm,
                    cudaStream_t stream,int warmup,int iterations,
                    std::ofstream& csv,const char* protocol) {
  if(bytes%sizeof(float)) MPI_Abort(MPI_COMM_WORLD,2);
  const size_t count=bytes/sizeof(float);
  float *sendbuf=nullptr,*recvbuf=nullptr;
  CUDA_CHECK(cudaMalloc(&sendbuf,bytes));
  CUDA_CHECK(cudaMalloc(&recvbuf,bytes));
  std::vector<float> input(count,float(rank+1)),output(count,0.0f);
  CUDA_CHECK(cudaMemcpy(sendbuf,input.data(),bytes,cudaMemcpyHostToDevice));

  for(int i=0;i<warmup;i++) launch(op,sendbuf,recvbuf,count,rank,comm,stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<double> apiSamples,e2eSamples;
  for(int i=0;i<iterations;i++) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); // outside measured interval
    auto e2eBegin=Clock::now();
    auto apiBegin=Clock::now();
    launch(op,sendbuf,recvbuf,count,rank,comm,stream);
    auto apiEnd=Clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto e2eEnd=Clock::now();

    double local[2]={elapsedUs(apiBegin,apiEnd),elapsedUs(e2eBegin,e2eEnd)};
    double rankMax[2]={0.0,0.0};
    MPI_CHECK(MPI_Allreduce(local,rankMax,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));
    apiSamples.push_back(rankMax[0]);
    e2eSamples.push_back(rankMax[1]);
  }

  CUDA_CHECK(cudaMemcpy(output.data(),recvbuf,bytes,cudaMemcpyDeviceToHost));
  int localOk=validate(op,output,rank)?1:0, allOk=0;
  MPI_CHECK(MPI_Allreduce(&localOk,&allOk,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD));
  if(!allOk) MPI_Abort(MPI_COMM_WORLD,3);

  const double apiMean=std::accumulate(apiSamples.begin(),apiSamples.end(),0.0)/iterations;
  const double e2eMean=std::accumulate(e2eSamples.begin(),e2eSamples.end(),0.0)/iterations;
  if(rank==0) {
    const char* name=op==Operation::AllReduce?"allreduce":"sendrecv";
    printf("%s,%s,%zu,%zu,%.3f,%.3f,%d,%d,OK\n",name,protocol,bytes,count,apiMean,e2eMean,warmup,iterations);
    csv<<name<<","<<protocol<<","<<bytes<<","<<count<<","<<apiMean<<","<<e2eMean<<","<<warmup<<","<<iterations<<",OK\n";
    csv.flush();
  }
  CUDA_CHECK(cudaFree(sendbuf));
  CUDA_CHECK(cudaFree(recvbuf));
}

int main(int argc,char** argv) {
  MPI_CHECK(MPI_Init(&argc,&argv));
  int rank=0,nranks=0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD,&nranks));
  if(nranks!=2) { if(rank==0) fprintf(stderr,"Require exactly 2 MPI ranks\n"); MPI_Abort(MPI_COMM_WORLD,1); }

  const int gpuBase=std::getenv("GPU_BASE")?std::atoi(std::getenv("GPU_BASE")):0;
  CUDA_CHECK(cudaSetDevice(gpuBase+rank));
  constexpr int warmup=20,iterations=100;
  const size_t sizes[]={8ULL,16ULL*1024ULL,8ULL*1024ULL*1024ULL};

  ncclUniqueId id;
  if(rank==0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm,nranks,id,rank));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));

  const char* protocol=std::getenv("NCCL_PROTO")?std::getenv("NCCL_PROTO"):"UNSET";
  const std::string csvPath=std::getenv("RESULT_CSV")?std::getenv("RESULT_CSV"):std::string("results_")+protocol+".csv";
  std::ofstream csv;
  if(rank==0) {
    csv.open(csvPath,std::ios::out|std::ios::trunc);
    if(!csv) MPI_Abort(MPI_COMM_WORLD,4);
    csv<<"operation,protocol,bytes,elements,host_api_mean_us,e2e_mean_us,warmup,iterations,validation\n";
    printf("operation,protocol,bytes,elements,host_api_mean_us,e2e_mean_us,warmup,iterations,validation\n");
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  for(size_t bytes:sizes) {
    runCase(Operation::AllReduce,bytes,rank,comm,stream,warmup,iterations,csv,protocol);
    runCase(Operation::SendRecv,bytes,rank,comm,stream,warmup,iterations,csv,protocol);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  CUDA_CHECK(cudaStreamDestroy(stream));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
