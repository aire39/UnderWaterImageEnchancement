#include <string_view>
#include <fstream>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot/implot.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>

#include "imageops/imageops.h"
#include "imageops/imagefilters.h"
#include "imageops/colormodel.h"

#define USE_ON_RESIZING true

std::pair<std::vector<uint8_t>, float> redefine_algo(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel);
std::vector<uint8_t> redefine(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel, float loss_limit);
std::vector<uint8_t> attenuation_map_max(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel);

int main(int argc, char*argv[])
{
  constexpr std::string_view window_name = "Underwater Image Enhancement";

  // setup command line arguments
  CLI::App app{"underwater image enhancement processing tool"};

  uint32_t window_width = 800;
  app.add_option("--ww", window_width, "set window width");

  uint32_t window_height = 600;
  app.add_option("--wh", window_height, "set window height");

  std::string image_file_path;
  app.add_option("-i,--image", image_file_path, "load image to process");

  CLI11_PARSE(app, argc, argv)

  // setup logger
  constexpr uint32_t number_of_backtrace_logs = 32;
  spdlog::enable_backtrace(number_of_backtrace_logs);

  // load image file and checkerboard if image is not found
  sf::Image loaded_image;

  constexpr uint32_t loaded_image_margin = 16;

  if (!image_file_path.empty())
  {
    spdlog::info("Trying to load: {}...", image_file_path);

    bool found_image = loaded_image.loadFromFile(image_file_path);
    if (!found_image)
    {
      spdlog::warn("Unable to find image file: {}", image_file_path);
    }
    else
    {
      // make the window twice as large of the original image and create pixels (margin) for some space
      // between the original and processed image
      window_width = (loaded_image.getSize().x * 2) + loaded_image_margin;
      window_height = loaded_image.getSize().y + loaded_image_margin;

      spdlog::info("loaded!");
    }
  }

  // setup sfml window
  spdlog::info("initializing window...");

  sf::RenderWindow window(sf::VideoMode({window_width, window_height})
                         ,window_name.data());

  // setup sprite

  sf::Texture loaded_texture;
  sf::Sprite loaded_image_plane;

  if (loaded_image.getSize() != sf::Vector2u(0.0f,0.0f))
  {
    spdlog::info("setup loaded image sprite...");
    loaded_texture.loadFromImage(loaded_image);
    loaded_image_plane.setTexture(loaded_texture);
    loaded_image_plane.setPosition(loaded_image_margin / 2.0f, loaded_image_margin / 2.0f);
  }

  // setup imgui
  spdlog::info("initializing ui...");
  
  sf::Clock delta_clock;
  bool imgui_initialized = ImGui::SFML::Init(window);
  ImPlot::CreateContext();

  if (!imgui_initialized)
  {
    spdlog::critical("failed to initialize ImGui!");
    abort();
  }

  // application loop
  spdlog::info("starting application loop...");

  #ifdef USE_SFML_ONRESIZING_EVENT
  #if USE_ON_RESIZING
  bool is_on_size_set = window.setOnSize([&](const sf::Event& event) {
    spdlog::info("window resizing");
  });

  if (is_on_size_set)
  {
    spdlog::info("custom window on resize (setOnSize) callback set!");
  }
  #endif
  #endif

  constexpr uint8_t bytes_per_pixel = 4;
  const uint32_t image_width = loaded_image.getSize().x;
  const uint32_t image_height = loaded_image.getSize().y;

  std::vector<uint8_t> input_image (loaded_image.getPixelsPtr()
                                   ,loaded_image.getPixelsPtr()+(image_width * image_height * bytes_per_pixel));

  auto input_image_split = imageops::channel_split(input_image.data(), image_width, image_height, bytes_per_pixel);
  auto input_image_split_red = imageops::convert_int_to_float_channel(input_image_split[0].data(), image_width, image_height);
  auto input_image_split_green = imageops::convert_int_to_float_channel(input_image_split[1].data(), image_width, image_height);
  auto input_image_split_blue = imageops::convert_int_to_float_channel(input_image_split[2].data(), image_width, image_height);

  // generate redefined images based on mean of channels

  constexpr float loss_limit = 1e-2f; // loss limit is set t0 10^(-2) as this gets us to the about the minimal amount of iterations needed for the redefine image channels to balance out
  std::vector<uint8_t> combined_channels_corrected_image = input_image;
  combined_channels_corrected_image = redefine(combined_channels_corrected_image, image_width, image_height, bytes_per_pixel, loss_limit);
  auto combined_channels_corrected_image_split = imageops::channel_split(combined_channels_corrected_image.data(), image_width, image_height, bytes_per_pixel);

  // generate attenuation channel (choose channel with the highest sum of pixel values)

  auto attenuation_channel = attenuation_map_max(input_image, image_width, image_height, bytes_per_pixel);
  auto normalized_attenuation_channel = imageops::normalize_channel(attenuation_channel.data(), image_width, image_height);
  auto expanded_attenuation_channel = imageops::expand_to_n_channels(attenuation_channel.data(), image_width, image_height, 1, bytes_per_pixel);

  // generate detailed image (un-sharpen filter per channel)

  std::vector<std::vector<uint8_t>> rgb_detail_mask(3, std::vector<uint8_t>(image_width * image_height));
  auto rgba_image_channels = imageops::channel_split(input_image.data(), image_width, image_height, bytes_per_pixel);
  constexpr float unsharp_const = 1.0f;
  auto sharpen_mask_red = imagefilters::unsharpen_channel(rgba_image_channels[0], image_width, image_height, unsharp_const);
  rgb_detail_mask[0] = imagefilters::constrain_filter_to_byte_map(sharpen_mask_red);

  auto sharpen_mask_green = imagefilters::unsharpen_channel(rgba_image_channels[1], image_width, image_height, unsharp_const);
  rgb_detail_mask[1] = imagefilters::constrain_filter_to_byte_map(sharpen_mask_green);

  auto sharpen_mask_blue = imagefilters::unsharpen_channel(rgba_image_channels[2], image_width, image_height, unsharp_const);
  rgb_detail_mask[2] = imagefilters::constrain_filter_to_byte_map(sharpen_mask_blue);

  auto expanded_byte_sharpen_mask_red = imageops::expand_to_n_channels(rgb_detail_mask[0].data(), image_width, image_height, 1, bytes_per_pixel);
  auto expanded_byte_sharpen_mask_green = imageops::expand_to_n_channels(rgb_detail_mask[1].data(), image_width, image_height, 1, bytes_per_pixel);
  auto expanded_byte_sharpen_mask_blue = imageops::expand_to_n_channels(rgb_detail_mask[2].data(), image_width, image_height, 1, bytes_per_pixel);

  // generate the Jaffe-McGlamey model --> J_c*t_c + A_c(1 - t_c), c E {R, G, B}
  // I_fc = D_c + I_ct*A_max + I_c*(1 - A_max)

  std::vector<float> neg_ones(image_width * image_height, 1.0f);
  auto jm_model_one_minus_a = imageops::element_subtract(neg_ones.data(), normalized_attenuation_channel.data(), image_width, image_height);

  auto combined_channels_corrected_image_split_red_float = imageops::convert_int_to_float_channel(combined_channels_corrected_image_split[0].data(), image_width, image_height);
  auto jm_model_red_jt = imageops::element_multi(combined_channels_corrected_image_split_red_float.data(), normalized_attenuation_channel.data(), image_width, image_height);
  auto jm_model_red_ai = imageops::element_multi(jm_model_one_minus_a.data(), input_image_split_red.data(), image_width, image_height);

  auto combined_channels_corrected_image_split_green_float = imageops::convert_int_to_float_channel(combined_channels_corrected_image_split[1].data(), image_width, image_height);
  auto jm_model_green_jt = imageops::element_multi(combined_channels_corrected_image_split_green_float.data(), normalized_attenuation_channel.data(), image_width, image_height);
  auto jm_model_green_ai = imageops::element_multi(jm_model_one_minus_a.data(), input_image_split_green.data(), image_width, image_height);

  auto combined_channels_corrected_image_split_blue_float = imageops::convert_int_to_float_channel(combined_channels_corrected_image_split[2].data(), image_width, image_height);
  auto jm_model_blue_jt = imageops::element_multi(combined_channels_corrected_image_split_blue_float.data(), normalized_attenuation_channel.data(), image_width, image_height);
  auto jm_model_blue_ai = imageops::element_multi(jm_model_one_minus_a.data(), input_image_split_blue.data(), image_width, image_height);

  auto jm_model_red = imageops::element_add(imageops::element_add(jm_model_red_jt.data(), jm_model_red_ai.data(), image_width, image_height).data(), sharpen_mask_red.data(), image_width, image_height);
  auto jm_model_green = imageops::element_add(imageops::element_add(jm_model_green_jt.data(), jm_model_green_ai.data(), image_width, image_height).data(), sharpen_mask_green.data(), image_width, image_height);
  auto jm_model_blue = imageops::element_add(imageops::element_add(jm_model_blue_jt.data(), jm_model_blue_ai.data(), image_width, image_height).data(), sharpen_mask_blue.data(), image_width, image_height);

  auto byte_jm_model_red = imagefilters::constrain_filter_to_byte_map(jm_model_red);
  auto byte_jm_model_green = imagefilters::constrain_filter_to_byte_map(jm_model_green);
  auto byte_jm_model_blue = imagefilters::constrain_filter_to_byte_map(jm_model_blue);

  std::vector<std::vector<uint8_t>> jm_model_channels(4);
  jm_model_channels[0] = byte_jm_model_red;
  jm_model_channels[1] = byte_jm_model_green;
  jm_model_channels[2] = byte_jm_model_blue;
  jm_model_channels[3] = rgba_image_channels[3];

  auto jm_model_color_corrected_image = imageops::channel_combine(jm_model_channels, image_width, image_height);

  // convert from rgb to cie-lab

  auto cielab_color_corrected_image = colormodel::convert_image_rgb_to_cielab(jm_model_color_corrected_image, image_width, image_height);
  auto cielab_color_corrected_image_split = imageops::channel_split(cielab_color_corrected_image.data(), image_width, image_height, bytes_per_pixel);

  auto cielab_rgb_color_corrected_image = colormodel::convert_image_cielab_to_rgb(cielab_color_corrected_image, image_width, image_height);
  auto cielab_rgb_color_corrected_image_split = imageops::channel_split(cielab_color_corrected_image.data(), image_width, image_height, bytes_per_pixel);

  // show original and resultant image

  sf::Image image_result;
  image_result.create(image_width, image_height, jm_model_color_corrected_image.data());

  std::string color_corrected_image_file_path = image_file_path.substr(0, image_file_path.find_last_of('.'));
  color_corrected_image_file_path += "_color_corrected.png";
  image_result.saveToFile(color_corrected_image_file_path);

  sf::Texture texture_result;
  texture_result.loadFromImage(image_result);
  sf::Sprite result_plane(texture_result);
  result_plane.setPosition(static_cast<float>(loaded_image.getSize().x), loaded_image_margin / 2.0f);

  while (window.isOpen())
  {
    ImGui::SFML::Update(window, delta_clock.restart());

    for (sf::Event event{}; window.pollEvent(event);)
    {
      ImGui::SFML::ProcessEvent(window, event);

      if (event.type == sf::Event::Closed)
      {
        window.close();
      }

      // catch the resize events
      if (event.type == sf::Event::Resized)
      {
        // update the view to the new size of the window
      }
    }

    // Render
    constexpr uint32_t cornflower_color = 0x9ACEEB;
    window.clear(sf::Color(cornflower_color));

    if (loaded_image.getSize() != sf::Vector2u(0,0))
    {
      window.draw(loaded_image_plane);
    }

    if (image_result.getSize() != sf::Vector2u(0,0))
    {
      window.draw(result_plane);
    }

    ImGui::SFML::Render(window);

    window.display();
  }

  ImPlot::DestroyContext();
  ImGui::SFML::Shutdown();

  spdlog::info("application done!");

  return 0;
}

std::pair<std::vector<uint8_t>, float> redefine_algo(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel)
{
  auto rgba_image_channels = imageops::channel_split(input_image.data(), image_width, image_height, bytes_per_pixel);

  float red_channel_mean = imageops::mean(rgba_image_channels[0].data(), image_width, image_height);
  float green_channel_mean = imageops::mean(rgba_image_channels[1].data(), image_width, image_height);
  float blue_channel_mean = imageops::mean(rgba_image_channels[2].data(), image_width, image_height);

  std::map<float, std::vector<uint8_t>> mean_channel_order;
  mean_channel_order[red_channel_mean] = rgba_image_channels[0];
  mean_channel_order[green_channel_mean] = rgba_image_channels[1];
  mean_channel_order[blue_channel_mean] = rgba_image_channels[2];

  float loss = std::numeric_limits<float>::max();
  float loss_1 = std::numeric_limits<float>::max();
  float loss_2 = std::numeric_limits<float>::max();

  size_t r_index = 0, g_index = 0, b_index = 0;

  std::vector<float> lms_mean;
  std::vector<std::pair<float, float>> lms_minmax;
  std::vector<std::vector<uint8_t>> lms_channels(3);
  size_t index = 3;
  for (auto [key, value] : mean_channel_order)
  {
    auto min_value = imageops::min_channel_value(value.data(), image_width, image_height);
    auto max_value = imageops::max_channel_value(value.data(), image_width, image_height);

    lms_minmax.insert(lms_minmax.begin(), {min_value, max_value});
    lms_mean.insert(lms_mean.begin(), key);
    lms_channels.insert(lms_channels.begin(), value);

    if (key == red_channel_mean)
    {
      r_index = index-- - 1;
    }

    if (key == green_channel_mean)
    {
      g_index = index-- - 1;
    }

    if (key == blue_channel_mean)
    {
      b_index = index-- - 1;
    }
  }

  // create correct image for each channel
  // calculate the loss values
  // find loss_color

  std::vector<std::vector<uint8_t>> corrected_lms_channel(3, std::vector<uint8_t>(image_width*image_height));

  constexpr float image_min_0 = 0.0f;
  constexpr float image_max_0 = 255.0f;
  for (size_t i=0; i<(image_width*image_height); i++)
  {
    const float range_minmax = ((image_max_0 - image_min_0) / (lms_minmax[0].second - lms_minmax[0].first));
    const float range = static_cast<float>(lms_channels[0][i]) - static_cast<float>(lms_minmax[0].first);
    float l_value = std::clamp(image_min_0 + (range * range_minmax), 0.0f, 255.0f);
    corrected_lms_channel[0][i] = static_cast<uint8_t>(l_value);
  }

  for (size_t i=0; i<(image_width*image_height); i++)
  {
    auto m_value = std::clamp(static_cast<float>(lms_channels[1][i]) + ((lms_mean[0] - lms_mean[1]) / 255.0f) * static_cast<float>(lms_channels[0][i]), 0.0f, 255.0f);
    corrected_lms_channel[1][i] = static_cast<uint8_t>(m_value);
  }

  for (size_t i=0; i<(image_width*image_height); i++)
  {
    auto s_value = std::clamp(static_cast<float>(lms_channels[2][i]) + ((lms_mean[1] - lms_mean[2]) / 255.0f) * static_cast<float>(lms_channels[1][i]), 0.0f, 255.0f);
    corrected_lms_channel[2][i] = static_cast<uint8_t>(s_value);
  }

  loss_1 = std::min(((lms_mean[0] - lms_mean[1]) / 255.0f), loss_1);
  loss_2 = std::min(((lms_mean[1] - lms_mean[2]) / 255.0f), loss_2);
  loss = std::min(std::abs(loss_1 - loss_2), loss);

  std::vector<std::vector<uint8_t>> corrected_images;
  corrected_images.emplace_back(corrected_lms_channel[r_index]);
  corrected_images.emplace_back(corrected_lms_channel[g_index]);
  corrected_images.emplace_back(corrected_lms_channel[b_index]);
  corrected_images.emplace_back(rgba_image_channels[3]);
  auto combined_channels_corrected_image = imageops::channel_combine(corrected_images, image_width, image_height);

  return {combined_channels_corrected_image, loss};
}

std::vector<uint8_t> redefine(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel, float loss_limit)
{
  float loss = 1.0f;
  std::vector<uint8_t> combined_channels_corrected_image = input_image;
  while (loss > loss_limit)
  {
    std::tie(combined_channels_corrected_image, loss) = redefine_algo(combined_channels_corrected_image, image_width, image_height, bytes_per_pixel);
  }

  return combined_channels_corrected_image;
}

std::vector<uint8_t> attenuation_map_max(const std::vector<uint8_t> & input_image, uint32_t image_width, uint32_t image_height, uint32_t bytes_per_pixel)
{
  constexpr float gamma = 1.2f; // controls intensity of received light
  auto rgba_image_channels = imageops::channel_split(input_image.data(), image_width, image_height, bytes_per_pixel);

  for (size_t i=0; i<image_height; i++)
  {
    for (size_t j=0; j<image_width; j++)
    {
      for (size_t k=0; k<3; k++)
      {
        float norm_pixel_value = static_cast<float>(rgba_image_channels[k][j + (i * image_width)]) / 255.0f;
        float new_pixel_value = 1.0f - std::pow(norm_pixel_value, gamma);
        rgba_image_channels[k][j + (i * image_width)] = static_cast<uint8_t>(new_pixel_value * 255.0f);
      }
    }
  }

  float r_max = imageops::channel_sum(rgba_image_channels[0].data(), image_width, image_height);
  float g_max = imageops::channel_sum(rgba_image_channels[1].data(), image_width, image_height);
  float b_max = imageops::channel_sum(rgba_image_channels[2].data(), image_width, image_height);

  std::vector<uint8_t> max_attenuation_channel;
  if ((r_max > g_max) && (r_max > b_max))
  {
    max_attenuation_channel = rgba_image_channels[0];
  }
  else if ((g_max > r_max) && (g_max > b_max))
  {
    max_attenuation_channel = rgba_image_channels[1];
  }
  else
  {
    max_attenuation_channel = rgba_image_channels[2];
  }

  return max_attenuation_channel;
}
