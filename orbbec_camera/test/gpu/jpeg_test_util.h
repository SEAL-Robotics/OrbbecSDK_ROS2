// Copyright 2026 SEAL Robotics
//
// libjpeg directly, for what OpenCV 4.6 does not expose: encoding with a
// chosen chroma subsampling (the camera sends 4:2:2) and decoding to the
// raw YCbCr planes.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

// After the C++ headers: libjpeg's header needs FILE and size_t.
#include <cstdio>
#include <jpeglib.h>

namespace orbbec_camera {
namespace gpu {
namespace test {

struct RawJpeg {
  int width = 0;
  int height = 0;
  int h_shift = 0;
  int v_shift = 0;
  std::vector<uint8_t> plane[3];
  size_t pitch[3] = {0, 0, 0};
};

inline RawJpeg decodeRaw(const std::vector<uint8_t> &jpeg) {
  jpeg_decompress_struct cinfo{};
  jpeg_error_mgr jerr{};
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, jpeg.data(), static_cast<unsigned long>(jpeg.size()));
  jpeg_read_header(&cinfo, TRUE);
  cinfo.raw_data_out = TRUE;
  jpeg_start_decompress(&cinfo);

  RawJpeg raw;
  raw.width = static_cast<int>(cinfo.image_width);
  raw.height = static_cast<int>(cinfo.image_height);
  raw.h_shift = cinfo.comp_info[0].h_samp_factor == 2 ? 1 : 0;
  raw.v_shift = cinfo.comp_info[0].v_samp_factor == 2 ? 1 : 0;
  int rows[3];
  for (int c = 0; c < 3; ++c) {
    const auto &comp = cinfo.comp_info[c];
    raw.pitch[c] = comp.width_in_blocks * DCTSIZE;
    rows[c] = comp.height_in_blocks * DCTSIZE + comp.v_samp_factor * DCTSIZE;
    raw.plane[c].assign(raw.pitch[c] * rows[c], 0);
  }
  const int mcu_rows = cinfo.max_v_samp_factor * DCTSIZE;
  std::vector<JSAMPROW> pointers[3];
  int row_base[3] = {0, 0, 0};
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPARRAY arrays[3];
    for (int c = 0; c < 3; ++c) {
      const int n = cinfo.comp_info[c].v_samp_factor * DCTSIZE;
      pointers[c].resize(n);
      for (int r = 0; r < n; ++r) {
        pointers[c][r] = raw.plane[c].data() + (row_base[c] + r) * raw.pitch[c];
      }
      arrays[c] = pointers[c].data();
      row_base[c] += n;
    }
    jpeg_read_raw_data(&cinfo, arrays, mcu_rows);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return raw;
}

// libjpeg encodes with the luma sampling factors asked for (OpenCV 4.6 has
// no switch for them): 2x1 is the camera's 4:2:2.
inline std::vector<uint8_t> encodeJpeg(const std::vector<uint8_t> &rgb, int width, int height,
                                       int h_samp, int v_samp, int quality) {
  jpeg_compress_struct cinfo{};
  jpeg_error_mgr jerr{};
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  unsigned char *out = nullptr;
  unsigned long out_size = 0;
  jpeg_mem_dest(&cinfo, &out, &out_size);
  cinfo.image_width = static_cast<JDIMENSION>(width);
  cinfo.image_height = static_cast<JDIMENSION>(height);
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);
  cinfo.comp_info[0].h_samp_factor = h_samp;
  cinfo.comp_info[0].v_samp_factor = v_samp;
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row = const_cast<JSAMPROW>(&rgb[cinfo.next_scanline * static_cast<size_t>(width) * 3]);
    jpeg_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  std::vector<uint8_t> jpeg(out, out + out_size);
  jpeg_destroy_compress(&cinfo);
  std::free(out);
  return jpeg;
}

}  // namespace test
}  // namespace gpu
}  // namespace orbbec_camera
