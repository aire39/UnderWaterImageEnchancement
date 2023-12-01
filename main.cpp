#include <string_view>
#include <thread>
#include <fstream>

#include <imgui.h>
#include <imgui-SFML.h>
#include <implot/implot.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>

#define USE_ON_RESIZING true

int main(int argc, char*argv[])
{
  constexpr std::string_view window_name = "Underwater Image Enchancement";

  // setup command line arguments
  CLI::App app{"underwater image enchancement processing tool"};

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
    window.clear(sf::Color::Blue);

    if (loaded_image.getSize() != sf::Vector2u(0,0))
    {
      window.draw(loaded_image_plane);
    }

    ImGui::SFML::Render(window);

    window.display();
  }

  ImPlot::DestroyContext();
  ImGui::SFML::Shutdown();

  spdlog::info("application done!");

  return 0;
}
