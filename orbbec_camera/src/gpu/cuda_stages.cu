// Copyright 2026 SEAL Robotics
// The alignment and remap kernels follow OrbbecSDK v2.10.6 AlignImpl.cpp and
// UnDistortionImplGeneric.cpp: Copyright (c) Orbbec Inc., licensed under the
// MIT License.

#include <cuda_runtime.h>

#include <cstring>
#include <sstream>
#include <vector>

#include "orbbec_camera/gpu/color_math.h"
#include "orbbec_camera/gpu/cuda_stages.h"

namespace orbbec_camera {
namespace gpu {
namespace {

constexpr int kBlock = 256;
constexpr int kInterBits = 5;
constexpr int kInterTab = 1 << kInterBits;

int gridFor(size_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

bool check(cudaError_t status, const char *what, std::string *error) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error) {
    std::ostringstream message;
    message << what << ": " << cudaGetErrorString(status);
    *error = message.str();
  }
  return false;
}

template <typename T>
struct DeviceBuffer {
  T *ptr = nullptr;
  size_t count = 0;

  ~DeviceBuffer() { cudaFree(ptr); }
  bool reserve(size_t n, std::string *error) {
    if (n <= count) {
      return true;
    }
    cudaFree(ptr);
    ptr = nullptr;
    count = 0;
    if (!check(cudaMalloc(&ptr, n * sizeof(T)), "cudaMalloc", error)) {
      return false;
    }
    count = n;
    return true;
  }
};

template <typename T>
struct PinnedBuffer {
  T *ptr = nullptr;
  size_t count = 0;

  ~PinnedBuffer() { cudaFreeHost(ptr); }
  bool reserve(size_t n, std::string *error) {
    if (n <= count) {
      return true;
    }
    cudaFreeHost(ptr);
    ptr = nullptr;
    count = 0;
    if (!check(cudaMallocHost(&ptr, n * sizeof(T)), "cudaMallocHost", error)) {
      return false;
    }
    count = n;
    return true;
  }
};

// A stream and a blocking-sync event: the host thread sleeps, rather than
// spins, until the GPU work it waits for is done.
struct StreamSync {
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;

  ~StreamSync() {
    if (done) {
      cudaEventDestroy(done);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  bool init(std::string *error) {
    if (stream) {
      return true;
    }
    return check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate",
                 error) &&
           check(cudaEventCreateWithFlags(&done, cudaEventBlockingSync | cudaEventDisableTiming),
                 "cudaEventCreate", error);
  }
  bool wait(std::string *error) {
    return check(cudaGetLastError(), "kernel launch", error) &&
           check(cudaEventRecord(done, stream), "cudaEventRecord", error) &&
           check(cudaEventSynchronize(done), "cudaEventSynchronize", error);
  }
};

// ---------------------------------------------------------------------------
// Speckle filter: union-find connected components over the 4-neighbourhood
// (Playne and Hawick, "A New Algorithm for Parallel Connected-Component
// Labelling on GPUs", 2018), then a size per root.

__device__ int findRoot(const volatile int *parent, int x) {
  int p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  return x;
}

__device__ void unite(int *parent, int a, int b) {
  bool done = false;
  do {
    a = findRoot(parent, a);
    b = findRoot(parent, b);
    if (a < b) {
      const int old = atomicMin(&parent[b], a);
      done = (old == b);
      b = old;
    } else if (b < a) {
      const int old = atomicMin(&parent[a], b);
      done = (old == a);
      a = old;
    } else {
      done = true;
    }
  } while (!done);
}

__global__ void speckleInit(const uint16_t *depth, int *parent, int *size, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  parent[i] = depth[i] != 0 ? i : -1;
  size[i] = 0;
}

__global__ void speckleMerge(const uint16_t *depth, int *parent, int width, int height,
                             int max_diff) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) {
    return;
  }
  const int d = depth[i];
  if (d == 0) {
    return;
  }
  const int x = i % width;
  const int y = i / width;
  if (x + 1 < width) {
    const int e = depth[i + 1];
    if (e != 0 && abs(d - e) <= max_diff) {
      unite(parent, i, i + 1);
    }
  }
  if (y + 1 < height) {
    const int s = depth[i + width];
    if (s != 0 && abs(d - s) <= max_diff) {
      unite(parent, i, i + width);
    }
  }
}

__global__ void speckleCount(int *parent, int *size, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || parent[i] < 0) {
    return;
  }
  const int root = findRoot(parent, i);
  parent[i] = root;
  atomicAdd(&size[root], 1);
}

__global__ void speckleApply(const uint16_t *depth, const int *parent, const int *size,
                             uint16_t *out, int n, int max_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int root = parent[i];
  out[i] = (root >= 0 && size[root] > max_size) ? depth[i] : 0;
}

// ---------------------------------------------------------------------------
// Depth-to-colour alignment, AlignImpl's SSE path: IEEE multiplies, adds and
// divides, fused only where the SDK's own build fuses, so every pixel lands
// where the SDK puts it.

struct AlignParams {
  float scaled_trans[3];
  float fx;
  float fy;
  float cx;
  float cy;
  int color_width;
  int color_height;
  unsigned int max_valid;
};

struct Projected {
  float x;
  float y;
  float z;
};

// sdk_algorithms.cpp mulAdd(): fused exactly where the SDK's build fuses.
__device__ float mulAdd(float a, float b, float c) {
#if defined(ORBBEC_SDK_FUSED_MULTIPLY_ADD)
  return __fmaf_rn(a, b, c);
#else
  return __fadd_rn(__fmul_rn(a, b), c);
#endif
}

__device__ Projected projectPixel(float d, float cxc, float cyc, float czc, const AlignParams &p) {
  const float y_cam = mulAdd(d, cyc, p.scaled_trans[1]);
  const float x_cam = mulAdd(d, cxc, p.scaled_trans[0]);
  const float z_cam = mulAdd(d, czc, p.scaled_trans[2]);
  const float nx = __fdiv_rn(x_cam, z_cam);
  const float ny = __fdiv_rn(y_cam, z_cam);
  return {mulAdd(nx, p.fx, p.cx), mulAdd(ny, p.fy, p.cy), z_cam};
}

__device__ bool truncFloat(float v, int &out) {
  if (!(v > -2147483648.0f && v < 2147483648.0f)) {
    return false;
  }
  out = __float2int_rz(v);
  return true;
}

__device__ bool truncDouble(double v, int &out) {
  if (!(v > -2147483648.0 && v < 2147483648.0)) {
    return false;
  }
  out = __double2int_rz(v);
  return true;
}

__device__ unsigned int depthToU16(float z) {
  if (!(z < 4294967296.0f)) {
    return 0xFFFFu;
  }
  return __float2uint_rz(z) & 0xFFFFu;
}

__global__ void alignFill(unsigned int *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 0xFFFFu;
  }
}

__global__ void alignSingle(const uint16_t *depth, const float *tx, const float *ty,
                            const float *tz, int n, AlignParams p, unsigned int *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int raw = depth[i];
  if (raw >= p.max_valid) {
    raw = 0;
  }
  const Projected q = projectPixel(static_cast<float>(raw), tx[i], ty[i], tz[i], p);
  if (static_cast<double>(q.z) < 1e-6) {
    return;
  }
  int u, v;
  if (!truncFloat(__fadd_rn(q.x, 0.5f), u) || !truncFloat(__fadd_rn(q.y, 0.5f), v)) {
    return;
  }
  if (u < 0 || u >= p.color_width || v < 0 || v >= p.color_height) {
    return;
  }
  const unsigned int cur = depthToU16(q.z);
  const int pos = v * p.color_width + u;
  atomicMin(&out[pos], cur);
  if (u + 1 < p.color_width) {
    atomicMin(&out[pos + 1], cur);
    if (v + 1 < p.color_height) {
      atomicMin(&out[pos + p.color_width + 1], cur);
    }
  }
  if (v + 1 < p.color_height) {
    atomicMin(&out[pos + p.color_width], cur);
  }
}

__global__ void alignMulti(const uint16_t *depth, const float *tx0, const float *ty0,
                           const float *tz0, const float *tx1, const float *ty1,
                           const float *tz1, int n, AlignParams p, unsigned int *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int raw = depth[i];
  if (raw >= p.max_valid) {
    raw = 0;
  }
  const float d = static_cast<float>(raw);
  const float *tx[2] = {tx0, tx1};
  const float *ty[2] = {ty0, ty1};
  const float *tz[2] = {tz0, tz1};
  int u[2];
  int v[2];
  unsigned int cur = 65535u;
  for (int c = 0; c < 2; ++c) {
    const Projected q = projectPixel(d, tx[c][i], ty[c][i], tz[c][i], p);
    if (static_cast<double>(q.z) < 1e-6) {
      return;
    }
    if (!truncDouble(__dadd_rn(static_cast<double>(q.x), 0.5), u[c]) ||
        !truncDouble(__dadd_rn(static_cast<double>(q.y), 0.5), v[c])) {
      return;
    }
    cur = min(cur, depthToU16(q.z));
  }
  const int u0 = max(0, min(u[0], u[1]));
  const int v0 = max(0, min(v[0], v[1]));
  const int u1 = min(max(u[0], u[1]), p.color_width - 1);
  const int v1 = min(max(v[0], v[1]), p.color_height - 1);
  for (int vr = v0; vr <= v1; ++vr) {
    for (int ur = u0; ur <= u1; ++ur) {
      atomicMin(&out[vr * p.color_width + ur], cur);
    }
  }
}

__global__ void alignFinish(const unsigned int *in, uint16_t *out, int n, unsigned int max_valid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const unsigned int v = in[i];
    out[i] = static_cast<uint16_t>(v >= max_valid ? 0u : v);
  }
}

// ---------------------------------------------------------------------------
// Undistortion remap, UnDistortionImplGeneric::remapNChannelBilinear.

__global__ void remapBilinear(const uint8_t *src, int channels, size_t src_pitch,
                              const int16_t *xi, const int16_t *yi, const uint16_t *alpha,
                              const uint8_t *oob, int width, int height, uint8_t *dst) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width * height) {
    return;
  }
  uint8_t *o = dst + static_cast<size_t>(i) * 3;
  if (oob[i]) {
    o[0] = o[1] = o[2] = 0;
    return;
  }
  const int x0 = xi[i];
  const int y0 = yi[i];
  const int a = alpha[i];
  const int fx = a & (kInterTab - 1);
  const int fy = a >> kInterBits;
  const int x1 = min(x0 + 1, width - 1);
  const int y1 = min(y0 + 1, height - 1);
  const int fxi = kInterTab - fx;
  const int fyi = kInterTab - fy;
  const int w00 = fxi * fyi;
  const int w10 = fx * fyi;
  const int w01 = fxi * fy;
  const int w11 = fx * fy;
  const uint8_t *row0 = src + static_cast<size_t>(y0) * src_pitch;
  const uint8_t *row1 = src + static_cast<size_t>(y1) * src_pitch;
  const uint8_t *p00 = row0 + x0 * channels;
  const uint8_t *p10 = row0 + x1 * channels;
  const uint8_t *p01 = row1 + x0 * channels;
  const uint8_t *p11 = row1 + x1 * channels;
  for (int c = 0; c < 3; ++c) {
    o[c] = static_cast<uint8_t>((w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c] + 512) >>
                                10);
  }
}

__global__ void ycbcrToRgb(YcbcrPlanes src, uint8_t *rgb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= src.width * src.height) {
    return;
  }
  ycbcrPixelToRgb(src, i % src.width, i / src.width, rgb + static_cast<size_t>(i) * 3);
}

__global__ void rgbToYuv420(const uint8_t *rgb, int width, int height, uint8_t *y,
                            size_t y_pitch, uint8_t *cb, size_t cb_pitch, uint8_t *cr,
                            size_t cr_pitch) {
  const int blocks_x = (width + 1) / 2;
  const int blocks_y = (height + 1) / 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks_x * blocks_y) {
    return;
  }
  const int bx = i % blocks_x;
  const int by = i / blocks_x;
  rgbBlockToYuv420(rgb, static_cast<size_t>(width) * 3, width, height, bx, by, y, y_pitch,
                   cb + by * cb_pitch + bx, cr + by * cr_pitch + bx);
}

}  // namespace

bool cudaDeviceAvailable(std::string *reason) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    if (reason) {
      *reason = std::string("no CUDA device: ") + cudaGetErrorString(status);
    }
    return false;
  }
  if (count == 0) {
    if (reason) {
      *reason = "no CUDA device";
    }
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------

struct CudaDepthStage::Impl {
  StreamSync sync;
  SpeckleParams speckle;
  bool has_alignment = false;
  D2CConfig config;
  int table_channels = 1;
  float scaled_trans[3] = {0.0f, 0.0f, 0.0f};
  DeviceBuffer<float> tables[6];
  DeviceBuffer<uint16_t> raw;
  DeviceBuffer<uint16_t> filtered;
  DeviceBuffer<int> parent;
  DeviceBuffer<int> size;
  DeviceBuffer<unsigned int> aligned32;
  DeviceBuffer<uint16_t> aligned;
  PinnedBuffer<uint16_t> staging_in;
  PinnedBuffer<uint16_t> staging_filtered;
  PinnedBuffer<uint16_t> staging_aligned;
};

CudaDepthStage::CudaDepthStage() : impl_(std::make_unique<Impl>()) {}
CudaDepthStage::~CudaDepthStage() = default;

void CudaDepthStage::setSpeckle(const SpeckleParams &params) { impl_->speckle = params; }

bool CudaDepthStage::hasAlignment() const { return impl_->has_alignment; }

bool CudaDepthStage::setAlignment(const D2CConfig &config, std::string *error) {
  Impl &s = *impl_;
  s.has_alignment = false;
  if (!s.sync.init(error)) {
    return false;
  }
  const D2CTables tables = buildD2CTables(config);
  const size_t n = static_cast<size_t>(tables.width) * tables.height;
  const std::vector<float> *host[6] = {&tables.x[0], &tables.y[0], &tables.z[0],
                                       &tables.x[1], &tables.y[1], &tables.z[1]};
  for (int k = 0; k < 3 * tables.channels; ++k) {
    if (!s.tables[k].reserve(n, error) ||
        !check(cudaMemcpy(s.tables[k].ptr, host[k]->data(), n * sizeof(float),
                          cudaMemcpyHostToDevice),
               "upload alignment table", error)) {
      return false;
    }
  }
  s.config = config;
  s.table_channels = tables.channels;
  for (int k = 0; k < 3; ++k) {
    s.scaled_trans[k] = tables.scaled_trans[k];
  }
  s.has_alignment = true;
  return true;
}

bool CudaDepthStage::process(const uint16_t *raw, int width, int height, bool speckle,
                             uint16_t *filtered_out, uint16_t *aligned_out, std::string *error) {
  Impl &s = *impl_;
  if (!s.sync.init(error)) {
    return false;
  }
  if (aligned_out && !s.has_alignment) {
    if (error) {
      *error = "alignment requested before setAlignment";
    }
    return false;
  }
  if (aligned_out &&
      (width != s.config.depth_intrinsic.width || height != s.config.depth_intrinsic.height)) {
    if (error) {
      std::ostringstream message;
      message << "depth is " << width << "x" << height << " but the alignment was set up for "
              << s.config.depth_intrinsic.width << "x" << s.config.depth_intrinsic.height;
      *error = message.str();
    }
    return false;
  }
  const int n = width * height;
  if (!s.raw.reserve(n, error) || !s.staging_in.reserve(n, error)) {
    return false;
  }
  std::memcpy(s.staging_in.ptr, raw, static_cast<size_t>(n) * sizeof(uint16_t));
  if (!check(cudaMemcpyAsync(s.raw.ptr, s.staging_in.ptr, n * sizeof(uint16_t),
                             cudaMemcpyHostToDevice, s.sync.stream),
             "upload depth", error)) {
    return false;
  }

  const uint16_t *depth = s.raw.ptr;
  if (speckle) {
    if (!s.filtered.reserve(n, error) || !s.parent.reserve(n, error) ||
        !s.size.reserve(n, error)) {
      return false;
    }
    speckleInit<<<gridFor(n), kBlock, 0, s.sync.stream>>>(s.raw.ptr, s.parent.ptr, s.size.ptr, n);
    speckleMerge<<<gridFor(n), kBlock, 0, s.sync.stream>>>(s.raw.ptr, s.parent.ptr, width,
                                                           height, s.speckle.max_diff);
    speckleCount<<<gridFor(n), kBlock, 0, s.sync.stream>>>(s.parent.ptr, s.size.ptr, n);
    speckleApply<<<gridFor(n), kBlock, 0, s.sync.stream>>>(s.raw.ptr, s.parent.ptr, s.size.ptr,
                                                           s.filtered.ptr, n, s.speckle.max_size);
    depth = s.filtered.ptr;
  }
  if (filtered_out) {
    if (!s.staging_filtered.reserve(n, error) ||
        !check(cudaMemcpyAsync(s.staging_filtered.ptr, depth, n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, s.sync.stream),
               "download filtered depth", error)) {
      return false;
    }
  }

  int m = 0;
  if (aligned_out) {
    const OBCameraIntrinsic &color = s.config.color_intrinsic;
    m = color.width * color.height;
    if (!s.aligned32.reserve(m, error) || !s.aligned.reserve(m, error) ||
        !s.staging_aligned.reserve(m, error)) {
      return false;
    }
    AlignParams p{};
    for (int k = 0; k < 3; ++k) {
      p.scaled_trans[k] = s.scaled_trans[k];
    }
    p.fx = color.fx;
    p.fy = color.fy;
    p.cx = color.cx;
    p.cy = color.cy;
    p.color_width = color.width;
    p.color_height = color.height;
    p.max_valid = s.config.max_valid_depth;
    alignFill<<<gridFor(m), kBlock, 0, s.sync.stream>>>(s.aligned32.ptr, m);
    if (s.table_channels == 1) {
      alignSingle<<<gridFor(n), kBlock, 0, s.sync.stream>>>(
          depth, s.tables[0].ptr, s.tables[1].ptr, s.tables[2].ptr, n, p, s.aligned32.ptr);
    } else {
      alignMulti<<<gridFor(n), kBlock, 0, s.sync.stream>>>(
          depth, s.tables[0].ptr, s.tables[1].ptr, s.tables[2].ptr, s.tables[3].ptr,
          s.tables[4].ptr, s.tables[5].ptr, n, p, s.aligned32.ptr);
    }
    alignFinish<<<gridFor(m), kBlock, 0, s.sync.stream>>>(s.aligned32.ptr, s.aligned.ptr, m,
                                                          p.max_valid);
    if (!check(cudaMemcpyAsync(s.staging_aligned.ptr, s.aligned.ptr, m * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, s.sync.stream),
               "download aligned depth", error)) {
      return false;
    }
  }

  if (!s.sync.wait(error)) {
    return false;
  }
  if (filtered_out) {
    std::memcpy(filtered_out, s.staging_filtered.ptr, static_cast<size_t>(n) * sizeof(uint16_t));
  }
  if (aligned_out) {
    std::memcpy(aligned_out, s.staging_aligned.ptr, static_cast<size_t>(m) * sizeof(uint16_t));
  }
  return true;
}

// ---------------------------------------------------------------------------

struct CudaColorStage::Impl {
  StreamSync sync;
  int width = 0;
  int height = 0;
  DeviceBuffer<int16_t> xi;
  DeviceBuffer<int16_t> yi;
  DeviceBuffer<uint16_t> alpha;
  DeviceBuffer<uint8_t> oob;
  DeviceBuffer<uint8_t> src;
  DeviceBuffer<uint8_t> rgb;
  PinnedBuffer<uint8_t> staging_in;
  PinnedBuffer<uint8_t> staging_out;
};

CudaColorStage::CudaColorStage() : impl_(std::make_unique<Impl>()) {}
CudaColorStage::~CudaColorStage() = default;

bool CudaColorStage::setLut(const UndistortLut &lut, std::string *error) {
  Impl &s = *impl_;
  s.width = 0;
  s.height = 0;
  if (!s.sync.init(error)) {
    return false;
  }
  const size_t n = static_cast<size_t>(lut.width) * lut.height;
  if (!s.xi.reserve(n, error) || !s.yi.reserve(n, error) || !s.alpha.reserve(n, error) ||
      !s.oob.reserve(n, error) || !s.rgb.reserve(n * 3, error)) {
    return false;
  }
  if (!check(cudaMemcpy(s.xi.ptr, lut.xi.data(), n * sizeof(int16_t), cudaMemcpyHostToDevice),
             "upload lut", error) ||
      !check(cudaMemcpy(s.yi.ptr, lut.yi.data(), n * sizeof(int16_t), cudaMemcpyHostToDevice),
             "upload lut", error) ||
      !check(cudaMemcpy(s.alpha.ptr, lut.alpha.data(), n * sizeof(uint16_t),
                        cudaMemcpyHostToDevice),
             "upload lut", error) ||
      !check(cudaMemcpy(s.oob.ptr, lut.oob.data(), n, cudaMemcpyHostToDevice), "upload lut",
             error)) {
    return false;
  }
  s.width = lut.width;
  s.height = lut.height;
  return true;
}

bool CudaColorStage::rectify(const DeviceImage &src, uint8_t *rgb_out, const Yuv420Planes *yuv_out,
                             std::string *error) {
  Impl &s = *impl_;
  if (s.width == 0) {
    if (error) {
      *error = "rectify called before setLut";
    }
    return false;
  }
  if (src.channels != 3 && src.channels != 4) {
    if (error) {
      *error = "source must have 3 or 4 channels";
    }
    return false;
  }
  const int n = s.width * s.height;
  remapBilinear<<<gridFor(n), kBlock, 0, s.sync.stream>>>(src.data, src.channels, src.pitch,
                                                          s.xi.ptr, s.yi.ptr, s.alpha.ptr,
                                                          s.oob.ptr, s.width, s.height, s.rgb.ptr);
  if (yuv_out) {
    const int blocks = ((s.width + 1) / 2) * ((s.height + 1) / 2);
    rgbToYuv420<<<gridFor(blocks), kBlock, 0, s.sync.stream>>>(
        s.rgb.ptr, s.width, s.height, yuv_out->plane[0], yuv_out->pitch[0], yuv_out->plane[1],
        yuv_out->pitch[1], yuv_out->plane[2], yuv_out->pitch[2]);
  }
  const size_t bytes = static_cast<size_t>(n) * 3;
  if (rgb_out) {
    if (!s.staging_out.reserve(bytes, error) ||
        !check(cudaMemcpyAsync(s.staging_out.ptr, s.rgb.ptr, bytes, cudaMemcpyDeviceToHost,
                               s.sync.stream),
               "download rgb", error)) {
      return false;
    }
  }
  if (!s.sync.wait(error)) {
    return false;
  }
  if (rgb_out) {
    std::memcpy(rgb_out, s.staging_out.ptr, bytes);
  }
  return true;
}

bool CudaColorStage::rectifyHost(const uint8_t *src, int channels, size_t pitch, uint8_t *rgb_out,
                                 const Yuv420Planes *yuv_out, std::string *error) {
  Impl &s = *impl_;
  if (s.width == 0) {
    if (error) {
      *error = "rectify called before setLut";
    }
    return false;
  }
  const size_t bytes = pitch * static_cast<size_t>(s.height);
  if (!s.src.reserve(bytes, error) || !s.staging_in.reserve(bytes, error)) {
    return false;
  }
  std::memcpy(s.staging_in.ptr, src, bytes);
  if (!check(cudaMemcpyAsync(s.src.ptr, s.staging_in.ptr, bytes, cudaMemcpyHostToDevice,
                             s.sync.stream),
             "upload colour", error)) {
    return false;
  }
  DeviceImage image;
  image.data = s.src.ptr;
  image.channels = channels;
  image.pitch = pitch;
  return rectify(image, rgb_out, yuv_out, error);
}

bool CudaColorStage::rectifyYcbcr(const YcbcrPlanes &src, uint8_t *rgb_out,
                                  const Yuv420Planes *yuv_out, std::string *error) {
  Impl &s = *impl_;
  if (s.width == 0) {
    if (error) {
      *error = "rectify called before setLut";
    }
    return false;
  }
  if (src.width != s.width || src.height != s.height) {
    if (error) {
      std::ostringstream message;
      message << "decoded frame is " << src.width << "x" << src.height << ", the LUT "
              << s.width << "x" << s.height;
      *error = message.str();
    }
    return false;
  }
  const size_t n = static_cast<size_t>(s.width) * s.height;
  if (!s.src.reserve(n * 3, error)) {
    return false;
  }
  ycbcrToRgb<<<gridFor(n), kBlock, 0, s.sync.stream>>>(src, s.src.ptr);
  DeviceImage image;
  image.data = s.src.ptr;
  image.channels = 3;
  image.pitch = static_cast<size_t>(s.width) * 3;
  return rectify(image, rgb_out, yuv_out, error);
}

}  // namespace gpu
}  // namespace orbbec_camera
