#pragma once

#include <cstdint>
#include <vector>

namespace imageops {
  float mean (const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float min_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float max_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float channel_sum(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  float channel_sum(const float * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_add(const uint8_t * image_data_channel_0, const uint8_t * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_add(const float * image_data_channel_0, const float * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_subtract(const uint8_t * image_data_channel_0, const uint8_t * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_subtract(const float * image_data_channel_0, const float * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_multi(const uint8_t * image_data_channel_0, const uint8_t * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> element_multi(const float * image_data_channel_0, const float * image_data_channel_1, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> normalize_channel(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<float> convert_int_to_float_channel(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<uint8_t> convert_float_to_int_channel(const float * image_data_channel, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<std::vector<uint8_t>> channel_split(const uint8_t * image_data, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & bpp);
  std::vector<std::vector<float>> channel_split(const float * image_data, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & bpp);
  std::vector<uint8_t> channel_combine(const std::vector<std::vector<uint8_t>> & image_channels, const uint32_t & image_width, const uint32_t & image_height);
  std::vector<uint8_t> expand_to_n_channels(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & input_bpp, const uint8_t & output_bpp);

  enum class CONV_TYPE : uint16_t {SUM=0, MULT, MIN, MAX, FRAC, POW};
  float image_convolution(const std::vector<uint8_t> & source
                         ,int32_t x
                         ,int32_t y
                         ,int32_t source_width
                         ,int32_t source_height
                         ,int32_t offset
                         ,int32_t sum_count
                         ,int32_t bpp
                         ,const std::vector<float> & kernel
                         ,int32_t kernel_width
                         ,int32_t kernel_height
                         ,float kernel_div
                         ,CONV_TYPE conv_type);
}
