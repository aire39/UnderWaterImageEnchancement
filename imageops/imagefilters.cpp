#include "imagefilters.h"

#include "imageops.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace imagefilters {

  std::vector<float> guassian_blur_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height)
  {
    std::vector<float> guassian_kernel = {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f , 1.0f, 2.0f, 1.0f};
    float guassian_kernel_divisor = 1.0f / imageops::channel_sum(guassian_kernel.data(), 3, 3);

    std::vector<float> guassian_blur_image (image_width * image_height);

    for (size_t i=0; i<image_height; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        constexpr int32_t offset = 0; // which channel to start or use for each pixel
        constexpr int32_t sum_count = 0; // can be used to sum components of pixel
        constexpr int32_t kernel_width = 3;
        constexpr int32_t kernel_height = 3;
        constexpr int32_t bpp = 1; // bytes per pixel

        float filter_value = imageops::image_convolution(input_image
                                                        ,j
                                                        ,i
                                                        ,image_width
                                                        ,image_height
                                                        ,offset
                                                        ,sum_count
                                                        ,bpp
                                                        ,guassian_kernel
                                                        ,kernel_width
                                                        ,kernel_height
                                                        ,guassian_kernel_divisor
                                                        ,imageops::CONV_TYPE::SUM);

        guassian_blur_image[j + (i * image_width)] = filter_value;
      }
    }

    return guassian_blur_image;

  }

  std::vector<float> unsharpen_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, const float & unsharp_const)
  {
    auto guassian_blur = guassian_blur_channel(input_image, image_width, image_height);

    std::vector<float> unsharp_mask_image (image_width * image_height);
    for (size_t i=0; i<image_height; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        unsharp_mask_image[j + (i * image_width)] = unsharp_const * (static_cast<float>(input_image[j + (i * image_width)]) - guassian_blur[j + (i * image_width)]);
      }
    }

    return unsharp_mask_image;
  }

  std::vector<float> integral_image_map(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height)
  {
    std::vector<float> integral_image (image_width * image_height);
    integral_image[0] = input_image[0];

    for (size_t i=1; i<image_width; i++)
    {
      integral_image[i] = integral_image[i-1] + input_image[i];
    }

    for (size_t i=1; i<image_height; i++)
    {
      float s = input_image[0 + (i * image_width)];
      integral_image[0 + (i * image_width)] = integral_image[0 + ((i - 1) * image_width)] + s;

      for (size_t j=1; j<image_width; j++)
      {
        s = s + input_image[j + (i * image_width)];
        integral_image[j + (i * image_width)] = integral_image[j + ((i - 1) * image_width)] + s;
      }
    }

    return integral_image;
  }

  float integral_image_map_local_block_mean(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height, uint32_t x, uint32_t y, uint32_t local_width, uint32_t local_height)
  {
    const size_t x_index_limit = (image_width - (x + local_width)) < local_width ? (image_width - (x + local_width)) : local_width;
    const size_t y_index_limit = (image_height - (y + local_height)) < local_height ? (image_height - (y + local_height)) : local_height;

    local_height = y_index_limit;
    local_width = x_index_limit;

    float l1 = ((x > 0) && (y > 0)) ? input_image[(x - 1) + ((y - 1) * image_width)] : 0.0f;
    float l2 = (y > 0) ? input_image[(x + local_width - 1) + ((y + 0) * image_width)] : 0.0f;
    float l3 = (x > 0) ? input_image[(x + 0) + ((y + local_height - 1) * image_width)] : 0.0f;
    float l4 = input_image[(x + local_width - 1) + ((y + local_height - 1) * image_width)];

    float mean = ((l4 + l1) - (l2 + l3)) / static_cast<float>(local_width * local_height);

    return mean;
  }

  float integral_image_map_local_block_variance(const std::vector<float> & input_image, uint32_t image_width, uint32_t image_height, uint32_t x, uint32_t y, uint32_t local_width, uint32_t local_height)
  {
    float l1 = ((x > 0) && (y > 0)) ? input_image[(x - 1) + ((y - 1) * image_width)] : 0.0f;
    float l2 = (y > 0) ? input_image[(x + local_width - 1) + ((y + 0) * image_width)] : 0.0f;
    float l3 = (x > 0) ? input_image[(x + 0) + ((y + local_height - 1) * image_width)] : 0.0f;
    float l4 = input_image[(x + local_width - 1) + ((y + local_height - 1) * image_width)];

    float mean = integral_image_map_local_block_mean(input_image, image_width, image_height, x, y, local_width, local_height);
    float variance = (std::pow(((l4 + l1) - (l2 + l3)), 2.0f) / static_cast<float>(local_width * local_height)) - std::pow(mean, 2.0f);

    return variance;
  }

  std::vector<uint8_t> constrain_filter_to_byte_map(const std::vector<float> & filter)
  {
    std::vector<uint8_t> constraint_result (filter.size());
    for (size_t i=0; i<filter.size(); i++)
    {
      constraint_result[i] = static_cast<uint8_t>(std::clamp(static_cast<float>(filter[i]), static_cast<float>(std::numeric_limits<uint8_t>::min()), static_cast<float>(std::numeric_limits<uint8_t>::max())));
    }

    return constraint_result;
  }

}