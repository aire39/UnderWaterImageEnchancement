#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace colormodel
{
  std::tuple<double , double, double> rgb2hsi(const uint8_t & r, const uint8_t & g, const uint8_t & b);
  std::tuple<double , double, double> rgb2hsl(const uint8_t & r, const uint8_t & g, const uint8_t & b);
  std::tuple<double , double, double> rgb2xyz(const uint8_t & r, const uint8_t & g, const uint8_t & b);
  std::tuple<uint8_t , uint8_t, uint8_t> xyz2rgb(const double & x, const double & y, const double & z);
  std::tuple<double , double, double> xyz2cielab(const double & x, const double & y, const double & z);
  std::tuple<double , double, double> cielab2xyz(const double &cie_l, const double & cie_a, const double & cie_b);
  std::tuple<double , double, double> rgb2cielab(const uint8_t & r, const uint8_t & g, const uint8_t & b);
  std::tuple<uint8_t , uint8_t, uint8_t> cielab2rgb(const uint8_t & r, const uint8_t & g, const uint8_t & b);
  std::tuple<uint8_t , uint8_t, uint8_t> hsi2rgb(const double & h, const double & s, const double & i);
  std::tuple<uint8_t , uint8_t, uint8_t> hsl2rgb(const double & h, const double & s, const double & l);

  std::vector<float> convert_image_rgb_to_cielab(std::vector<uint8_t> input_image, uint32_t image_width, uint32_t image_height);
  std::vector<uint8_t> convert_image_cielab_to_rgb(std::vector<float> input_image, uint32_t image_width, uint32_t image_height);
}
