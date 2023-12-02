#pragma once

#include <cstdint>
#include <vector>

namespace imagefilters {

  std::vector<float> guassian_blur_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height);
  std::vector<float> unsharpen_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height);
  std::vector<uint8_t> constrain_filter_to_byte_map(const std::vector<float> & filter);
};


