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
}
