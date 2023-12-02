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
        avg += static_cast<float>(image_data_channel[j + (i*image_width)]);
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

  float channel_sum(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    float sum = 0.0f;
    for (size_t i=0; i<(image_width * image_height); i++)
    {
      sum += static_cast<float>(image_data_channel[i]);
    }

    return sum;
  }

  float channel_sum(const float * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    float sum = 0.0f;
    for (size_t i=0; i<(image_width * image_height); i++)
    {
      sum += image_data_channel[i];
    }

    return sum;
  }

  std::vector<float> normalize_channel(const uint8_t * image_data_channel, const uint32_t & image_width, const uint32_t & image_height)
  {
    std::vector<float> normalized_channel (image_width * image_height);
    for (size_t i=0; i< (image_width * image_height); i++)
    {
      normalized_channel[i] = static_cast<float>(image_data_channel[i]) / static_cast<float>(std::numeric_limits<uint8_t>::max());
    }

    return  normalized_channel;
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
                        ,CONV_TYPE conv_type)
  {
    float value = 0.0;

    if (conv_type == CONV_TYPE::MULT)
    {
      value = 1.0f;
    }
    else if (conv_type == CONV_TYPE::MIN)
    {
      value = 255.0f;
    }

    int32_t k_width_centered = (kernel_width - 1) / 2;
    int32_t k_height_centered = (kernel_height - 1) / 2;

    int32_t convo_height_it_begin = y - k_height_centered;
    int32_t convo_height_it_end = (y + k_height_centered) + 1;
    int32_t convo_height_diff = (convo_height_it_end - convo_height_it_begin);
    convo_height_it_end = std::clamp(convo_height_it_end, 0, source_height);

    int32_t convo_width_it_begin = x - k_width_centered;
    int32_t convo_width_it_end = (x + k_width_centered) + 1;
    int32_t convo_width_diff = (convo_width_it_end - convo_width_it_begin);
    convo_width_it_end = std::clamp(convo_width_it_end, 0, source_width);

    for (int32_t i=convo_height_it_begin; i<convo_height_it_end; i++)
    {
      for (int32_t j=convo_width_it_begin; j<convo_width_it_end; j++)
      {
        int32_t kernel_x_index = (kernel_width - convo_width_diff) + (j - convo_width_it_begin);
        int32_t kernel_y_index = (kernel_height - convo_height_diff) + (i - convo_height_it_begin);

        int32_t jj = std::clamp(j, 0, source_width);
        int32_t ii = std::clamp(i, 0, source_height);

        if (sum_count > 0)
        {
          float sum_value = 0.0f;
          for (int32_t k=0; k<sum_count; k++)
          {
            sum_value += static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset + k]);
          }
          sum_value /= static_cast<float>(sum_count);

          if (conv_type == CONV_TYPE::SUM)
          {
            value += (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
          }
          else if (conv_type == CONV_TYPE::MIN)
          {
            auto tmp = (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
            value = (value > tmp) ? tmp : value;
          }
          else if (conv_type == CONV_TYPE::MAX)
          {
            auto tmp = (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
            value = (value < tmp) ? tmp : value;
          }
          else if (conv_type == CONV_TYPE::FRAC)
          {
            auto tmp = (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
            value += (kernel_div / tmp);
          }
          else if (conv_type == CONV_TYPE::POW)
          {
            auto tmp = (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
            value += std::pow(tmp, kernel_div);
          }
          else // CONV_TYPE::MULT
          {
            value *= (sum_value * static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
          }
        }
        else
        {
          if (conv_type == CONV_TYPE::SUM)
          {
            value += (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                      static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
          }
          else if (conv_type == CONV_TYPE::MIN)
          {
            float tmp = (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                         static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));

            value = (value > tmp) ? tmp : value;
          }
          else if (conv_type == CONV_TYPE::MAX)
          {
            float tmp = (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                         static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));

            value = (value < tmp) ? tmp : value;
          }
          else if (conv_type == CONV_TYPE::FRAC)
          {
            float tmp = (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                         static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));

            value += (kernel_div / tmp);
          }
          else if (conv_type == CONV_TYPE::POW)
          {
            float tmp = (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                         static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));

            value += std::pow(tmp, kernel_div);
          }
          else // CONV_TYPE::MULT
          {
            value *= (static_cast<float>(source[(jj * bpp) + (ii * source_width * bpp) + offset]) *
                      static_cast<float>(kernel[kernel_x_index + (kernel_y_index * kernel_width)]));
          }
        }
      }
    }

    float final_value = value;

    if (conv_type == CONV_TYPE::SUM)
    {
      final_value = (value * kernel_div);
    }

    return final_value;
  }
}