#pragma once

#include <cstdint>
#include <vector>

namespace imageops {
  float mean (const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float min_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float max_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<std::vector<uint8_t>> channel_split(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & bpp);
  std::vector<uint8_t> channel_combine(const std::vector<std::vector<uint8_t>> & image_channels, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<uint8_t> expand_to_n_channels(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & input_bpp, const uint8_t & output_bpp);
}
