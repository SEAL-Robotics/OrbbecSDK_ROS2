// Copyright 2026 SEAL Robotics
// The decode follows this package's jetson_nv_decoder.cpp
// (Copyright (c) 2023 Orbbec 3D Technology, Inc, Apache License 2.0) and the
// encode the Jetson Multimedia API's NvJpegEncoder::encodeFromFd.

#include "nv_jpeg.h"

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <setjmp.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>

// After the system headers: libjpeg's headers need FILE and size_t.
#include <jpeglib.h>
#include <nvbufsurface.h>

#include "jpegint.h"

namespace orbbec_camera {
namespace gpu {
namespace {

// libnvmm_jpeg exports the standard libjpeg names, at libjpeg 9's ABI, while
// OpenCV and the SDK in the same process use libjpeg 8 under the same names.
// Linked normally, whichever loads first would serve both. Opened local and
// called through this table it serves only this file, and its DT_SYMBOLIC
// keeps its own internal calls inside it.
struct LibNvJpeg {
  decltype(&jpeg_std_error) std_error = nullptr;
  decltype(&jpeg_CreateDecompress) create_decompress = nullptr;
  decltype(&jpeg_destroy_decompress) destroy_decompress = nullptr;
  decltype(&jpeg_mem_src) mem_src = nullptr;
  decltype(&jpeg_read_header) read_header = nullptr;
  decltype(&jpeg_start_decompress) start_decompress = nullptr;
  decltype(&jpeg_read_raw_data) read_raw_data = nullptr;
  decltype(&jpeg_finish_decompress) finish_decompress = nullptr;
  decltype(&jpeg_CreateCompress) create_compress = nullptr;
  decltype(&jpeg_destroy_compress) destroy_compress = nullptr;
  decltype(&jpeg_suppress_tables) suppress_tables = nullptr;
  decltype(&jpeg_mem_dest) mem_dest = nullptr;
  decltype(&jpeg_set_defaults) set_defaults = nullptr;
  decltype(&jpeg_set_quality) set_quality = nullptr;
  decltype(&jpeg_set_hardware_acceleration_parameters_enc) set_hw_enc = nullptr;
  decltype(&jpeg_start_compress) start_compress = nullptr;
  decltype(&jpeg_write_raw_data) write_raw_data = nullptr;
  decltype(&jpeg_finish_compress) finish_compress = nullptr;

  static const LibNvJpeg *get(std::string *error) {
    static std::once_flag once;
    static LibNvJpeg lib;
    static std::string load_error;
    std::call_once(once, [] { load_error = lib.load(); });
    if (!load_error.empty()) {
      *error = load_error;
      return nullptr;
    }
    return &lib;
  }

 private:
  template <typename F>
  bool bind(void *handle, const char *name, F *fn, std::string *missing) {
    *fn = reinterpret_cast<F>(dlsym(handle, name));
    if (*fn == nullptr) {
      *missing = name;
    }
    return *fn != nullptr;
  }

  std::string load() {
    void *handle = dlopen("libnvmm_jpeg.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char *reason = dlerror();
      return std::string("dlopen libnvmm_jpeg.so: ") + (reason ? reason : "unknown error");
    }
    std::string missing;
    const bool ok = bind(handle, "jpeg_std_error", &std_error, &missing) &&
                    bind(handle, "jpeg_CreateDecompress", &create_decompress, &missing) &&
                    bind(handle, "jpeg_destroy_decompress", &destroy_decompress, &missing) &&
                    bind(handle, "jpeg_mem_src", &mem_src, &missing) &&
                    bind(handle, "jpeg_read_header", &read_header, &missing) &&
                    bind(handle, "jpeg_start_decompress", &start_decompress, &missing) &&
                    bind(handle, "jpeg_read_raw_data", &read_raw_data, &missing) &&
                    bind(handle, "jpeg_finish_decompress", &finish_decompress, &missing) &&
                    bind(handle, "jpeg_CreateCompress", &create_compress, &missing) &&
                    bind(handle, "jpeg_destroy_compress", &destroy_compress, &missing) &&
                    bind(handle, "jpeg_suppress_tables", &suppress_tables, &missing) &&
                    bind(handle, "jpeg_mem_dest", &mem_dest, &missing) &&
                    bind(handle, "jpeg_set_defaults", &set_defaults, &missing) &&
                    bind(handle, "jpeg_set_quality", &set_quality, &missing) &&
                    bind(handle, "jpeg_set_hardware_acceleration_parameters_enc", &set_hw_enc,
                         &missing) &&
                    bind(handle, "jpeg_start_compress", &start_compress, &missing) &&
                    bind(handle, "jpeg_write_raw_data", &write_raw_data, &missing) &&
                    bind(handle, "jpeg_finish_compress", &finish_compress, &missing);
    return ok ? std::string() : "libnvmm_jpeg.so lacks " + missing;
  }
};

struct JpegErrorManager {
  jpeg_error_mgr base;
  jmp_buf jump_buffer;
  char message[JMSG_LENGTH_MAX];
};

void jpegErrorExit(j_common_ptr cinfo) {
  auto *error = reinterpret_cast<JpegErrorManager *>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, error->message);
  longjmp(error->jump_buffer, 1);
}

// libjpeg's warnings (corrupt-data notices among them) go to stderr by
// default; the caller sees failures through the return value instead.
void jpegQuiet(j_common_ptr, int) {}

std::string cuError(CUresult status, const char *what) {
  const char *name = nullptr;
  cuGetErrorString(status, &name);
  std::ostringstream message;
  message << what << ": " << (name ? name : "unknown CUDA driver error");
  return message.str();
}

// A pitch-linear hardware surface mapped for CUDA through EGL; registered
// once and kept, so a frame costs no mapping.
struct CudaSurface {
  NvBufSurface *surface = nullptr;
  CUgraphicsResource resource = nullptr;
  CUeglFrame frame{};

  ~CudaSurface() {
    if (resource) {
      cuGraphicsUnregisterResource(resource);
    }
    if (surface) {
      NvBufSurfaceUnMapEglImage(surface, 0);
      NvBufSurfaceDestroy(surface);
    }
  }

  bool create(int width, int height, NvBufSurfaceColorFormat format, std::string *error) {
    NvBufSurfaceCreateParams params{};
    params.gpuId = 0;
    params.width = static_cast<uint32_t>(width);
    params.height = static_cast<uint32_t>(height);
    params.colorFormat = format;
    params.layout = NVBUF_LAYOUT_PITCH;
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    if (NvBufSurfaceCreate(&surface, 1, &params) != 0) {
      surface = nullptr;
      *error = "NvBufSurfaceCreate failed";
      return false;
    }
    surface->numFilled = 1;
    if (NvBufSurfaceMapEglImage(surface, 0) != 0 ||
        surface->surfaceList[0].mappedAddr.eglImage == nullptr) {
      *error = "NvBufSurfaceMapEglImage failed";
      return false;
    }
    CUresult status = cuGraphicsEGLRegisterImage(
        &resource, static_cast<EGLImageKHR>(surface->surfaceList[0].mappedAddr.eglImage),
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS) {
      resource = nullptr;
      *error = cuError(status, "cuGraphicsEGLRegisterImage");
      return false;
    }
    status = cuGraphicsResourceGetMappedEglFrame(&frame, resource, 0, 0);
    if (status != CUDA_SUCCESS) {
      *error = cuError(status, "cuGraphicsResourceGetMappedEglFrame");
      return false;
    }
    if (frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
      *error = "surface did not map as pitch-linear memory";
      return false;
    }
    return true;
  }

  int fd() const { return static_cast<int>(surface->surfaceList[0].bufferDesc); }
  size_t pitch(int plane) const { return surface->surfaceList[0].planeParams.pitch[plane]; }
  uint8_t *plane(int index) const { return static_cast<uint8_t *>(frame.frame.pPitch[index]); }
};

}  // namespace

struct NvJpegHardware::Impl {
  const LibNvJpeg *lib = nullptr;
  int width = 0;
  int height = 0;

  jpeg_decompress_struct dinfo{};
  JpegErrorManager derr{};
  bool decoder_ready = false;

  jpeg_compress_struct cinfo{};
  JpegErrorManager cerr{};
  bool encoder_ready = false;

  CudaSurface yuv;
  // The decoded frame's planes as CUDA sees them, mapped per frame: the
  // decoder owns the surface and may replace it.
  NvBufSurface *decoded = nullptr;
  bool decoded_egl_mapped_here = false;
  CUgraphicsResource decoded_resource = nullptr;
  Yuv420Planes yuv_planes;
  std::vector<unsigned char> out_buffer;
  // A member, not a local: it must survive a longjmp out of the encoder.
  unsigned char *out = nullptr;
  // The decoded frame's sampling, read before jpeg_finish_decompress frees
  // comp_info.
  int h_samp = 0;
  int v_samp = 0;

  ~Impl() {
    // The EGL resources need a current CUDA context on the tearing-down thread.
    cudaFree(nullptr);
    releaseDecoded();
    destroyDecoder();
    if (encoder_ready) {
      lib->destroy_compress(&cinfo);
    }
  }

  void releaseDecoded() {
    if (decoded_resource) {
      cuGraphicsUnregisterResource(decoded_resource);
      decoded_resource = nullptr;
    }
    if (decoded && decoded_egl_mapped_here) {
      NvBufSurfaceUnMapEglImage(decoded, 0);
    }
    decoded = nullptr;
    decoded_egl_mapped_here = false;
  }

  bool createDecoder() {
    std::memset(&dinfo, 0, sizeof(dinfo));
    std::memset(&derr, 0, sizeof(derr));
    dinfo.err = lib->std_error(&derr.base);
    derr.base.error_exit = jpegErrorExit;
    derr.base.emit_message = jpegQuiet;
    if (setjmp(derr.jump_buffer) != 0) {
      return false;
    }
    lib->create_decompress(&dinfo, JPEG_LIB_VERSION, sizeof(dinfo));
    dinfo.mjpeg_decode = TRUE;
    decoder_ready = true;
    return true;
  }

  void destroyDecoder() {
    if (decoder_ready) {
      lib->destroy_decompress(&dinfo);
      decoder_ready = false;
    }
  }

  bool createEncoder() {
    std::memset(&cinfo, 0, sizeof(cinfo));
    std::memset(&cerr, 0, sizeof(cerr));
    cinfo.err = lib->std_error(&cerr.base);
    cerr.base.error_exit = jpegErrorExit;
    cerr.base.emit_message = jpegQuiet;
    if (setjmp(cerr.jump_buffer) != 0) {
      return false;
    }
    lib->create_compress(&cinfo, JPEG_LIB_VERSION, sizeof(cinfo));
    lib->suppress_tables(&cinfo, TRUE);
    encoder_ready = true;
    return true;
  }

  // NVJPG decode into the decoder's own hardware surface; returns its fd.
  bool decodeToFd(const uint8_t *jpeg, size_t size, int *fd, std::string *error) {
    if (!decoder_ready && !createDecoder()) {
      *error = std::string("NVJPG decoder init: ") + derr.message;
      return false;
    }
    derr.message[0] = '\0';
    if (setjmp(derr.jump_buffer) != 0) {
      *error = std::string("NVJPG decode: ") + derr.message;
      // The object is unusable after a longjmp out of libjpeg.
      destroyDecoder();
      return false;
    }
    NvBufSurface surface;
    dinfo.out_color_space = JCS_YCbCr;
    lib->mem_src(&dinfo, jpeg, size);
    (void)lib->read_header(&dinfo, TRUE);
    dinfo.out_color_space = JCS_YCbCr;
    dinfo.IsVendorbuf = TRUE;
    dinfo.pVendor_buf = reinterpret_cast<unsigned char *>(&surface);
    lib->start_decompress(&dinfo);
    if (dinfo.global_state != DSTATE_READY) {
      *error = "NVJPG decode: decompressor not ready after start";
      destroyDecoder();
      return false;
    }
    h_samp = dinfo.comp_info[0].h_samp_factor;
    v_samp = dinfo.comp_info[0].v_samp_factor;
    lib->read_raw_data(&dinfo, nullptr, v_samp * DCTSIZE);
    lib->finish_decompress(&dinfo);
    if (static_cast<int>(dinfo.image_width) != width ||
        static_cast<int>(dinfo.image_height) != height) {
      std::ostringstream message;
      message << "JPEG is " << dinfo.image_width << "x" << dinfo.image_height << ", expected "
              << width << "x" << height;
      *error = message.str();
      return false;
    }
    *fd = dinfo.fd;
    return true;
  }

  int hShift() const { return h_samp == 2 ? 1 : 0; }
  int vShift() const { return v_samp == 2 ? 1 : 0; }
};

NvJpegHardware::NvJpegHardware() : impl_(std::make_unique<Impl>()) {}
NvJpegHardware::~NvJpegHardware() = default;

std::unique_ptr<NvJpegHardware> NvJpegHardware::create(int width, int height,
                                                       std::string *error) {
  // The runtime makes the device's primary context current on this thread,
  // which the driver-API EGL calls below need.
  if (cudaFree(nullptr) != cudaSuccess) {
    *error = "no CUDA context";
    return nullptr;
  }
  const LibNvJpeg *lib = LibNvJpeg::get(error);
  if (lib == nullptr) {
    return nullptr;
  }
  std::unique_ptr<NvJpegHardware> hw(new NvJpegHardware());
  Impl &s = *hw->impl_;
  s.lib = lib;
  s.width = width;
  s.height = height;
  if (!s.createDecoder()) {
    *error = std::string("NVJPG decoder init: ") + s.derr.message;
    return nullptr;
  }
  if (!s.createEncoder()) {
    *error = std::string("NVJPG encoder init: ") + s.cerr.message;
    return nullptr;
  }
  std::string surface_error;
  if (!s.yuv.create(width, height, NVBUF_COLOR_FORMAT_YUV420_ER, &surface_error)) {
    *error = "hardware surface: " + surface_error;
    return nullptr;
  }
  for (int k = 0; k < 3; ++k) {
    s.yuv_planes.plane[k] = s.yuv.plane(k);
    s.yuv_planes.pitch[k] = s.yuv.pitch(k);
  }
  s.out_buffer.resize(static_cast<size_t>(width) * height * 3 / 2);
  return hw;
}

bool NvJpegHardware::decode(const uint8_t *jpeg, size_t size, YcbcrPlanes *planes,
                            std::string *error) {
  Impl &s = *impl_;
  s.releaseDecoded();
  int fd = -1;
  if (!s.decodeToFd(jpeg, size, &fd, error)) {
    return false;
  }
  if (s.v_samp > s.h_samp) {
    *error = "JPEG chroma subsampled only vertically, which this path does not convert";
    return false;
  }
  if (NvBufSurfaceFromFd(fd, reinterpret_cast<void **>(&s.decoded)) != 0 || !s.decoded) {
    s.decoded = nullptr;
    *error = "NvBufSurfaceFromFd failed for the decoded frame";
    return false;
  }
  NvBufSurfaceParams &surface = s.decoded->surfaceList[0];
  if (surface.layout != NVBUF_LAYOUT_PITCH || surface.planeParams.num_planes != 3) {
    *error = "decoded frame is not three pitch-linear planes";
    s.decoded = nullptr;
    return false;
  }
  if (surface.mappedAddr.eglImage == nullptr) {
    if (NvBufSurfaceMapEglImage(s.decoded, 0) != 0 || surface.mappedAddr.eglImage == nullptr) {
      *error = "NvBufSurfaceMapEglImage failed for the decoded frame";
      s.decoded = nullptr;
      return false;
    }
    s.decoded_egl_mapped_here = true;
  }
  CUresult status = cuGraphicsEGLRegisterImage(
      &s.decoded_resource, static_cast<EGLImageKHR>(surface.mappedAddr.eglImage),
      CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
  if (status != CUDA_SUCCESS) {
    s.decoded_resource = nullptr;
    *error = cuError(status, "cuGraphicsEGLRegisterImage (decoded frame)");
    s.releaseDecoded();
    return false;
  }
  CUeglFrame frame{};
  status = cuGraphicsResourceGetMappedEglFrame(&frame, s.decoded_resource, 0, 0);
  if (status != CUDA_SUCCESS || frame.frameType != CU_EGL_FRAME_TYPE_PITCH ||
      frame.planeCount != 3) {
    *error = status != CUDA_SUCCESS ? cuError(status, "cuGraphicsResourceGetMappedEglFrame")
                                    : "decoded frame did not map as three pitch-linear planes";
    s.releaseDecoded();
    return false;
  }
  for (int k = 0; k < 3; ++k) {
    planes->plane[k] = static_cast<const uint8_t *>(frame.frame.pPitch[k]);
    planes->pitch[k] = surface.planeParams.pitch[k];
  }
  planes->width = s.width;
  planes->height = s.height;
  planes->h_shift = s.hShift();
  planes->v_shift = s.vShift();
  return true;
}

void NvJpegHardware::releaseDecoded() { impl_->releaseDecoded(); }

const Yuv420Planes &NvJpegHardware::encodeInput() const { return impl_->yuv_planes; }

bool NvJpegHardware::encode(int quality, std::vector<uint8_t> *jpeg, std::string *error) {
  Impl &s = *impl_;
  if (!s.encoder_ready && !s.createEncoder()) {
    *error = std::string("NVJPG encoder init: ") + s.cerr.message;
    return false;
  }
  s.out = s.out_buffer.data();
  size_t out_size = s.out_buffer.size();
  s.cerr.message[0] = '\0';
  if (setjmp(s.cerr.jump_buffer) != 0) {
    *error = std::string("NVJPG encode: ") + s.cerr.message;
    s.lib->destroy_compress(&s.cinfo);
    s.encoder_ready = false;
    if (s.out != s.out_buffer.data()) {
      std::free(s.out);
    }
    return false;
  }
  s.lib->mem_dest(&s.cinfo, &s.out, &out_size);
  s.cinfo.fd = s.yuv.fd();
  s.cinfo.IsVendorbuf = TRUE;
  s.cinfo.raw_data_in = TRUE;
  s.cinfo.in_color_space = JCS_YCbCr;
  s.lib->set_defaults(&s.cinfo);
  s.lib->set_quality(&s.cinfo, quality, TRUE);
  s.lib->set_hw_enc(&s.cinfo, TRUE, static_cast<unsigned int>(out_size), 0, 0);
  s.cinfo.in_color_space = JCS_YCbCr;
  s.lib->start_compress(&s.cinfo, 0);
  s.lib->write_raw_data(&s.cinfo, nullptr, 0);
  s.lib->finish_compress(&s.cinfo);
  jpeg->assign(s.out, s.out + out_size);
  // libjpeg moves to a buffer of its own when ours is too small.
  if (s.out != s.out_buffer.data()) {
    std::free(s.out);
  }
  return true;
}

}  // namespace gpu
}  // namespace orbbec_camera
