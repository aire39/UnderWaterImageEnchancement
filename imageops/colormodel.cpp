#include "colormodel.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace {
  double deg2rad(double deg)
  {
    return deg * std::numbers::pi / 180.0;
  }

  double rad2deg(double rad)
  {
    return rad * 180.0 / std::numbers::pi;
  }
}

namespace colormodel
{
  std::tuple<double , double, double> rgb2hsi(const uint8_t & r, const uint8_t & g, const uint8_t & b)
  {
    double r_norm = static_cast<double>(r) / static_cast<double>(r + g + b);
    double g_norm = static_cast<double>(g) / static_cast<double>(r + g + b);
    double b_norm = static_cast<double>(b) / static_cast<double>(r + g + b);

    const double i = (static_cast<double>(r) + static_cast<double>(g) + static_cast<double>(b)) / (3.0 * 255.0);
    double h;

    if (g >= b)
    {
      h = std::acos((r_norm - (0.5*g_norm) - (0.5*b_norm)) / std::sqrt((r_norm*r_norm) + (g_norm*g_norm) + (b_norm*b_norm) - (r_norm*g_norm) - (r_norm*b_norm) - (g_norm*b_norm)));
      h = std::isnan(h) ? 0 : h;
    }
    else
    {
      h = deg2rad(360.0) - std::acos((r_norm - (0.5*g_norm) - (0.5*b_norm)) / std::sqrt((r_norm*r_norm) + (g_norm*g_norm) + (b_norm*b_norm) - (r_norm*g_norm) - (r_norm*b_norm) - (g_norm*b_norm)));
      h = std::isnan(h) ? 0 : h;
    }

    double s = 0.0f;
    if (i > 0)
    {
      s = static_cast<double>(1.0 - ((std::min(std::min(r, g), b)) / 255.0) / i);
    }

    constexpr double h_bias = 0.000;
    return {rad2deg(h + h_bias), (s * 100.0), std::clamp(i * 255.0, 0.0, 255.0)};
  }

  std::tuple<double , double, double> rgb2hsl(const uint8_t & r, const uint8_t & g, const uint8_t & b)
  {
    double rr = r / 255.0;
    double gg = g / 255.0;
    double bb = b / 255.0;

    const double c_min = std::min(std::min(rr, gg), bb);
    const double c_max = std::max(std::max(rr, gg), bb);
    const double c_delta = (c_max - c_min);

    const double l = (c_max + c_min) / 2.0;
    double h = 0.0f;
    double s = 0.0f;

    if (c_delta == 0)
    {
      h = 0;
    }
    else
    {
      if (c_max == rr)
      {
        double aa0 = (gg - bb) / c_delta;
        double aa1 = static_cast<double>(static_cast<int32_t>(aa0 * 100000) % static_cast<int32_t>(6.0 * 100000)) / 100000.0;
        h = 60 * aa1;
      }
      else if (c_max == gg)
      {
        h = 60 * ((static_cast<int32_t>((static_cast<double>(bb * 100000) - static_cast<double>(rr * 100000)) / static_cast<double>(c_delta)) + (2 * 100000)) / 100000.0);
      }
      else // (c_max == bb)
      {
        h = 60 * ((static_cast<int32_t>((static_cast<double>(rr * 100000) - static_cast<double>(gg * 100000)) / static_cast<double>(c_delta)) + (4 * 100000) ) / 100000.0);
      }
    }

    if (c_delta == 0);
    else
    {
      s = c_delta / (1.0 - std::abs((2 * l) - 1.0));
    }

    h = h < 0 ? std::round((h + 360) + 0.001) : std::round(h);

    return {h, (s * 100.0), (l * 100.0)};
  }

  std::tuple<double , double, double> rgb2xyz(const uint8_t & r, const uint8_t & g, const uint8_t & b)
  {
    //D65/2 reference
    double r_norm = static_cast<double>(r) / 255.0;
    double g_norm = static_cast<double>(g) / 255.0;
    double b_norm = static_cast<double>(b) / 255.0;

    if (r_norm > 0.04045)
    {
      r_norm = std::pow((r_norm + 0.055) / 1.055, 2.4);
    }
    else
    {
      r_norm /= 12.92;
    }

    if (g_norm > 0.04045)
    {
      g_norm = std::pow((g_norm + 0.055) / 1.055, 2.4);
    }
    else
    {
      g_norm /= 12.92;
    }

    if (b_norm > 0.04045)
    {
      b_norm = std::pow((b_norm + 0.055) / 1.055, 2.4);
    }
    else
    {
      b_norm /= 12.92;
    }

    r_norm *= 100.0;
    g_norm *= 100.0;
    b_norm *= 100.0;

    double x = (r_norm * 0.4124) + (g_norm * 0.3576) + (b_norm * 0.1805);
    double y = (r_norm * 0.2126) + (g_norm * 0.7152) + (b_norm * 0.0722);
    double z = (r_norm * 0.0193) + (g_norm * 0.1192) + (b_norm * 0.9505);

    return {x, y, z};
  }

  std::tuple<uint8_t , uint8_t, uint8_t> xyz2rgb(const double & x, const double & y, const double & z)
  {
    double var_x = static_cast<double>(x) / 100.0;
    double var_y = static_cast<double>(y) / 100.0;
    double var_z = static_cast<double>(z) / 100.0;

    double var_r = (var_x * 3.2406) + (var_y * -1.5372) + (var_z * -0.4986);
    double var_g = (var_x * -0.9689) + (var_y * 1.8758) + (var_z * 0.0415);
    double var_b = (var_x * 0.0557) + (var_y * -0.2040) + (var_z * 1.0570);

    if (var_r > 0.0031308)
    {
      var_r = 1.055 * std::pow(var_r, 1.0 / 2.4) - 0.055;
    }
    else
    {
      var_r *= 12.92;
    }

    if (var_g > 0.0031308)
    {
      var_g = 1.055 * std::pow(var_g, 1.0 / 2.4) - 0.055;
    }
    else
    {
      var_g *= 12.92;
    }

    if (var_b > 0.0031308)
    {
      var_b = 1.055 * std::pow(var_b, 1.0 / 2.4) - 0.055;
    }
    else
    {
      var_b *= 12.92;
    }

    auto r = static_cast<uint8_t>(var_r * 255.0);
    auto g = static_cast<uint8_t>(var_g * 255.0);
    auto b = static_cast<uint8_t>(var_b * 255.0);

    return {r, g, b};
  }

  std::tuple<double , double, double> xyz2cielab(const double & x, const double & y, const double & z)
  {
    // using the D65/2 xyz reference values
    //94.416 100.000 120.641 : D75
    double var_x = x / 94.416;
    double var_y = y / 100.0;
    double var_z = z / 120.641;

    if (var_x > 0.008856)
    {
      var_x = std::pow(var_x , 1.0/3.0);
    }
    else
    {
      var_x = (var_x * 7.787) + (16.0/116.0);
    }

    if (var_y > 0.008856)
    {
      var_y = std::pow(var_y , 1.0/3.0);
    }
    else
    {
      var_y = (var_y * 7.787) + (16.0/116.0);
    }

    if (var_z > 0.008856)
    {
      var_z = std::pow(var_z , 1.0/3.0);
    }
    else
    {
      var_z = (var_z * 7.787) + (16.0/116.0);
    }

    double cie_l = (116.0 * var_y) - 16.0;
    double cie_a = 500.0 * (var_x - var_y);
    double cie_b = 200.0 * (var_y - var_z);

    return {cie_l, cie_a, cie_b};
  }

  std::tuple<double , double, double> cielab2xyz(const double &cie_l, const double & cie_a, const double & cie_b)
  {
    double var_y = (cie_l + 16.0) / 116.0;
    double var_x = (cie_a / 500) + var_y;
    double var_z = var_y - (cie_b / 200);

    if (std::pow(var_y, 3.0) > 0.008856)
    {
      var_y = std::pow(var_y , 3.0);
    }
    else
    {
      var_y = (var_y - (16.0 / 116.0)) / 7.787;
    }

    if (std::pow(var_x, 3.0) > 0.008856)
    {
      var_x = std::pow(var_x , 3.0);
    }
    else
    {
      var_x = (var_x - (16.0 / 116.0)) / 7.787;
    }

    if (std::pow(var_z, 3.0) > 0.008856)
    {
      var_z = std::pow(var_z , 3.0);
    }
    else
    {
      var_z = (var_z - (16.0 / 116.0)) / 7.787;
    }

    // using the D65/2 xyz reference values
    // 95.047 100.000 108.883 : D65/2
    //94.416 100.000 120.641 : D75

    double x = var_x * 94.416f;
    double y = var_y * 100.0f;
    double z = var_z * 120.641f;

    return {x, y, z};
  }

  std::tuple<double , double, double> rgb2cielab(const uint8_t & r, const uint8_t & g, const uint8_t & b)
  {
    auto [x, y, z] = rgb2xyz(r, g, b);
    auto [cie_l, cie_a, cie_b] = xyz2cielab(x, y, z);
    return {cie_l, cie_a, cie_b};
  }

  std::tuple<uint8_t , uint8_t, uint8_t> cielab2rgb(const double & cie_l, const double & cie_a, const double & cie_b)
  {
    auto [x, y, z] = cielab2xyz(cie_l, cie_a, cie_b);
    auto [r, g, b] = xyz2rgb(x, y, z);
    return {r, g, b};
  }

  std::tuple<uint8_t , uint8_t, uint8_t> hsi2rgb(const double & h, const double & s, const double & i)
  {
    uint8_t r = 0, g = 0, b = 0;

    double sat = static_cast<double>(s) / 100.0f;
    double hue = deg2rad(static_cast<double>(h));
    auto intensity = static_cast<double>(i) / 255.0;

    if (std::floor(h) == 0)
    {
      r = static_cast<uint8_t>(std::round((intensity + (2.0 * intensity * sat)) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
    }
    else if (std::floor(h) > 0 && std::floor(h) < 120)
    {
      r = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * std::cos(hue) / std::cos(deg2rad(60.0) - hue)) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * (1.0 - std::cos(hue) / std::cos(deg2rad(60.0) - hue))) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
    }
    else if (std::floor(h) == 120)
    {
      r = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity + (2.0 * intensity * sat)) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
    }
    else if (std::floor(h) > 120 && std::floor(h) < 240)
    {
      r = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * std::cos(hue - deg2rad(120.0)) / std::cos(deg2rad(180.0) - hue)) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * (1.0 - std::cos(hue - deg2rad(120.0)) / std::cos(deg2rad(180.0) - hue))) * 255.0));
    }
    else if (std::abs(h) == 240)
    {
      r = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity + (2.0 * intensity * sat)) * 255.0));
    }
    else if (std::floor(h) > 240 && std::floor(h) < 360)
    {
      r = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * (1.0 - std::cos(hue - deg2rad(240.0)) / std::cos(deg2rad(300.0) - hue))) * 255.0));
      g = static_cast<uint8_t>(std::round((intensity - (intensity * sat)) * 255.0));
      b = static_cast<uint8_t>(std::round((intensity + (intensity * sat) * std::cos(hue - deg2rad(240.0)) / std::cos(deg2rad(300.0) - hue)) * 255.0));
    }

    return {r, g, b};
  }

  std::tuple<uint8_t , uint8_t, uint8_t> hsl2rgb(const double & h, const double & s, const double & l)
  {
    uint8_t r = 0, g = 0, b = 0;

    const double c = (1.0 - std::abs(2.0 * (l / 100.0) - 1.0)) * (s / 100.0);

    const double n_h = (h / 60.0);
    const double n_h_mod = static_cast<double>(static_cast<int32_t>(n_h * 100000) % static_cast<int32_t>(2.0 * 100000)) / 100000.0;
    const double x = c * (1.0 - std::abs(n_h_mod - 1.0));

    const double m = (l / 100.0) - (c / 2.0);

    double r_convert, g_convert, b_convert;


    if ( 0 <= h && h < 60)
    {
      r_convert = c;
      g_convert = x;
      b_convert = 0;
    }
    else if (60 <= h && h < 120)
    {
      r_convert = x;
      g_convert = c;
      b_convert = 0;
    }
    else if (120 <= h && h < 180)
    {
      r_convert = 0;
      g_convert = c;
      b_convert = x;
    }
    else if (180 <= h && h < 240)
    {
      r_convert = 0;
      g_convert = x;
      b_convert = c;
    }
    else if (240 <= h && h < 300)
    {
      r_convert = x;
      g_convert = 0;
      b_convert = c;
    }
    else // (300 <= h && h < 360)
    {
      r_convert = c;
      g_convert = 0;
      b_convert = x;
    }

    r = static_cast<uint8_t>((r_convert + m) * 255.0);
    g = static_cast<uint8_t>((g_convert + m) * 255.0);
    b = static_cast<uint8_t>((b_convert + m) * 255.0);

    return {r, g, b};
  }

  std::vector<float> convert_image_rgb_to_cielab(std::vector<uint8_t> input_image, uint32_t image_width, uint32_t image_height)
  {
    std::vector<float> cie_lab(image_width * image_height * 4, 100.0f);
    for(size_t i=0; i<image_height; i++)
    {
      for (int j = 0; j < image_width; ++j)
      {
        auto red_value = input_image[(j * 4) + (i * image_width * 4) + 0];
        auto green_value = input_image[(j * 4) + (i * image_width * 4) + 1];
        auto blue_value = input_image[(j * 4) + (i * image_width * 4) + 2];
        auto [cie_l, cie_a, cie_b] = rgb2cielab(red_value, green_value, blue_value);

        cie_lab[(j * 4) + (i * image_width * 4) + 0] = static_cast<float>(cie_l);
        cie_lab[(j * 4) + (i * image_width * 4) + 1] = static_cast<float>(cie_a);
        cie_lab[(j * 4) + (i * image_width * 4) + 2] = static_cast<float>(cie_b);
      }
    }

    return cie_lab;
  }

  std::vector<uint8_t> convert_image_cielab_to_rgb(std::vector<float> input_image, uint32_t image_width, uint32_t image_height)
  {
    std::vector<uint8_t> rgba(image_width * image_height * 4, 255);
    for(size_t i=0; i<image_height; i++)
    {
      for (int j = 0; j < image_width; ++j)
      {
        auto cie_l_value = input_image[(j * 4) + (i * image_width * 4) + 0];
        auto cie_a_value = input_image[(j * 4) + (i * image_width * 4) + 1];
        auto cie_b_value = input_image[(j * 4) + (i * image_width * 4) + 2];
        auto [r, g, b] = cielab2rgb(cie_l_value, cie_a_value, cie_b_value);

        rgba[(j * 4) + (i * image_width * 4) + 0] = r;
        rgba[(j * 4) + (i * image_width * 4) + 1] = g;
        rgba[(j * 4) + (i * image_width * 4) + 2] = b;
      }
    }

    return rgba;
  }

}
