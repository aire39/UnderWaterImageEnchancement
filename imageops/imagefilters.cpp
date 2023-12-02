#include "imagefilters.h"

#include "imageops.h"
#include <algorithm>
#include <limits>

namespace imagefilters {

  std::vector<float> guassian_blur_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height)
  {
    std::vector<float> guassian_kernel = {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f , 1.0f, 2.0f, 1.0f};
    float guassian_kernel_divisor = 1.0f / imageops::channel_sum(guassian_kernel.data(), 3, 3);

    std::vector<float> guassian_blur_image (image_width * image_height);

    for (size_t i=0; i<image_width; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        constexpr int32_t offset = 0;
        constexpr int32_t sum_count = 0;
        constexpr int32_t kernel_width = 3;
        constexpr int32_t kernel_height = 3;
        constexpr int32_t bpp = 1;

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

  std::vector<float> unsharpen_channel(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height)
  {
    constexpr float unsharp_const = -1.0f;
    auto guassian_blur = guassian_blur_channel(input_image, image_width, image_height);

    std::vector<float> unsharp_mask_image (image_width * image_height);
    for (size_t i=0; i<image_width; i++)
    {
      for (size_t j=0; j<image_width; j++)
      {
        unsharp_mask_image[j + (i * image_width)] = unsharp_const * static_cast<float>(input_image[j + (i * image_width)]) - guassian_blur[j + (i * image_width)];
      }
    }

    return unsharp_mask_image;
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