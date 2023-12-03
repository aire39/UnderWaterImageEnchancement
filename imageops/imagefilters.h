#pragma once

#include <cstdint>
#include <vector>

namespace imagefilters {

  std::vector<float> guassian_blur_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height);
  std::vector<float> unsharpen_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, const float & unsharp_const);

  std::vector<float> integral_image_map(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height);
  float integral_image_map_local_block_mean(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height, uint32_t x, uint32_t y, uint32_t local_width, uint32_t local_height);
  float integral_image_map_local_block_variance(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height, uint32_t x, uint32_t y, uint32_t local_width, uint32_t local_height);

  std::vector<uint8_t> constrain_filter_to_byte_map(const std::vector<float> & filter);
};


