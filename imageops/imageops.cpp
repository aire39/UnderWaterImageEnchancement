#include "imageops.h"

#include <limits>
#include <spdlog/spdlog.h>

namespace imageops {
  float mean (const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    float avg = 0.0f;

    for (size_t i=0; i<image_height; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        avg += image_data_channel[j + (i*image_width)];
      }
    }

    avg /= static_cast<float>(image_width * image_height);

    return avg;
  }

  float min_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    auto value = static_cast<float>(std::numeric_limits<uint8_t>::max());
    for (size_t i=0; i<image_height; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        value = std::min(value, static_cast<float>(image_data_channel[j + (i*image_width)]));
      }
    }

    return value;
  }

  float max_channel_value(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    auto value = static_cast<float>(std::numeric_limits<uint8_t>::min());
    for (size_t i=0; i<image_height; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        value = std::max(value, static_cast<float>(image_data_channel[j + (i*image_width)]));
      }
    }

    return value;
  }

  std::vector<std::vector<uint8_t>> channel_split(const uint8_t * image_data, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & bpp)
  {
    std::vector<std::vector<uint8_t>> channels;
    for (size_t i=0; i<bpp; i++)
    {
      channels.emplace_back(image_width*image_height);
    }

    for (size_t i=0; i<(image_width * image_height * bpp); i++)
    {
      channels[(i % bpp)][(i / bpp)] = image_data[i];
    }

    return channels;
  }

  std::vector<uint8_t> channel_combine(const std::vector<std::vector<uint8_t>> & image_channels, const uint32_t & image_width, const uint32_t & image_height)
  {
    const size_t n_channels = image_channels.size();
    std::vector<uint8_t> combined_channels (image_width * image_height * n_channels);

    for (size_t i=0; i<image_height; i++)
    {
      for(size_t j=0; j<image_width; j++)
      {
        for (size_t k=0; k<image_channels.size(); k++)
        {
          combined_channels[(j * n_channels + k) + (i * image_width * n_channels)] = image_channels[k][j + (i * image_width)];
        }
      }
    }

    return combined_channels;
  }

  std::vector<uint8_t> expand_to_n_channels(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height, const uint8_t & input_bpp, const uint8_t & output_bpp)
  {
    std::vector<uint8_t> channel (image_width * image_height * output_bpp, std::numeric_limits<uint8_t>::max());

    for (size_t i=0; i<(image_width * image_height); i++)
    {
      for (size_t j=0; j<(output_bpp - input_bpp); j++)
      {
        channel[(i * output_bpp) + j] = image_data_channel[(i * input_bpp) + (j / (output_bpp - input_bpp))];
      }
    }

    return channel;
  }
}