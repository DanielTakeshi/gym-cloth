#include "viewer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "console.h"

#if defined(NANOGUI_GLAD)
#if defined(NANOGUI_SHARED) && !defined(GLAD_GLAPI_EXPORT)
#define GLAD_GLAPI_EXPORT
#endif

#include <glad/glad.h>
#else
#if defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#else
#define GL_GLEXT_PROTOTYPES
#endif
#endif

#include <GLFW/glfw3.h>
#include <nanogui/nanogui.h>

using namespace std;
using namespace chrono;
using namespace nanogui;

#define DEFAULT_W 800
#define DEFAULT_H 600

namespace CGL {

// HDPI display
bool Viewer::HDPI;

// framecount & related timeers
int Viewer::framecount;
time_point<system_clock> Viewer::sys_last;
time_point<system_clock> Viewer::sys_curr;

// draw toggles
bool Viewer::showInfo = true;

// window properties
GLFWwindow *Viewer::window;
size_t Viewer::buffer_w;
size_t Viewer::buffer_h;

// user space renderer
Renderer *Viewer::renderer;

// GUI toolkit
Screen *Viewer::screen;

Viewer::Viewer() {}

Viewer::~Viewer() {

  glfwDestroyWindow(window);
  glfwTerminate();
  // free resources
  delete renderer;
}

void Viewer::init() {
  nanogui::init();

  // // initialize glfw
  // glfwSetErrorCallback(err_callback);
  // if (!glfwInit()) {
  //   out_err("Error: could not initialize GLFW!");
  //   exit(1);
  // }
  //
  // glfwSetTime(0);
  //
  // glfwWindowHint(GLFW_SAMPLES, 0);
  // glfwWindowHint(GLFW_RED_BITS, 8);
  // glfwWindowHint(GLFW_GREEN_BITS, 8);
  // glfwWindowHint(GLFW_BLUE_BITS, 8);
  // glfwWindowHint(GLFW_ALPHA_BITS, 8);
  // glfwWindowHint(GLFW_STENCIL_BITS, 8);
  // glfwWindowHint(GLFW_DEPTH_BITS, 24);
  // glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  //
  // // create window
  // string title = renderer ? "CGL: " + renderer->name() : "CGL";
  // window = glfwCreateWindow(DEFAULT_W, DEFAULT_H, title.c_str(), NULL, NULL);
  // if (!window) {
  //   out_err("Error: could not create window!");
  //   glfwTerminate();
  //   exit(1);
  // }
  //
  // // set context
  // glfwMakeContextCurrent(window);
  // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  //   throw std::runtime_error("Could not initialize GLAD!");
  // glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
  //
  // // Create a nanogui screen and pass the glfw pointer to initialize
  // screen = new Screen();
  // screen->initialize(window, true);
  //
  // int width, height;
  //
  // glfwGetFramebufferSize(window, &width, &height);
  // glViewport(0, 0, width, height);
  // glfwSwapInterval(0);
  // glfwSwapBuffers(window);

  // // framebuffer event callbacks
  // glfwSetFramebufferSizeCallback(window, resize_callback);
  //
  // // key event callbacks
  // glfwSetKeyCallback(window, key_callback);
  //
  // // cursor event callbacks
  // glfwSetCursorPosCallback(window, cursor_callback);
  //
  // // wheel event callbacks
  // glfwSetScrollCallback(window, scroll_callback);
  //
  // // mouse button callbacks
  // glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
  // glfwSetMouseButtonCallback(window, mouse_button_callback);

  // enable alpha blending
  // glEnable(GL_BLEND);
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  {
    // renderer->drawAll();
    // renderer->setVisible(true);

    nanogui::mainloop();
  }

  FormHelper *gui = new FormHelper(screen);

  // resize components to current window size, get DPI
  // glfwGetFramebufferSize(window, (int *)&buffer_w, (int *)&buffer_h);
  // if (buffer_w > DEFAULT_W)
  //   HDPI = true;
  //
  // // initialize renderer if already set
  // if (renderer) {
  //   if (HDPI)
  //     renderer->use_hdpi_render_target();
  renderer->init();
  // }

  bool enabled = true;
  FormHelper *gui = new FormHelper(screen);
  nanogui::ref<Window> nanoguiWindow =
      gui->addWindow(Eigen::Vector2i(10, 10), "Form helper example");

  gui->addGroup("Other widgets");
  gui->addButton("A button",
                 []() { std::cout << "Button pressed." << std::endl; });

  screen->setVisible(true);
  screen->performLayout();
  nanoguiWindow->center();

  // resize elements to current size
  resize_callback(window, buffer_w, buffer_h);
}

void Viewer::start() {

  // start timer
  sys_last = system_clock::now();

  // run update loop
  while (!glfwWindowShouldClose(window)) {
    update();
  }
}

void Viewer::set_renderer(Renderer *renderer) { this->renderer = renderer; }

void Viewer::update() {
  // poll events
  glfwPollEvents();

  glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // run user renderer
  if (renderer) {
    renderer->render();
  }

  screen->drawContents();
  screen->drawWidgets();

  // swap buffers
  glfwSwapBuffers(window);
}

void Viewer::err_callback(int error, const char *description) {
  out_err("GLFW Error: " << description);
}

void Viewer::resize_callback(GLFWwindow *window, int width, int height) {

  // get framebuffer size
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);

  // update buffer size
  buffer_w = w;
  buffer_h = h;
  glViewport(0, 0, buffer_w, buffer_h);

  // resize render if there is a user space renderer
  if (renderer) {
    renderer->resize(buffer_w, buffer_h);
  }

  // screen->resizeCallbackEvent(width, height);
}

void Viewer::cursor_callback(GLFWwindow *window, double xpos, double ypos) {
  // forward pan event to renderer
  if (HDPI) {
    float cursor_x = 2 * xpos;
    float cursor_y = 2 * ypos;
    renderer->cursor_event(cursor_x, cursor_y);
  } else {
    float cursor_x = xpos;
    float cursor_y = ypos;
    renderer->cursor_event(cursor_x, cursor_y);
  }

  screen->cursorPosCallbackEvent(xpos, ypos);
}

void Viewer::scroll_callback(GLFWwindow *window, double xoffset,
                             double yoffset) {

  renderer->scroll_event(xoffset, yoffset);
  screen->scrollCallbackEvent(xoffset, yoffset);
}

void Viewer::mouse_button_callback(GLFWwindow *window, int button, int action,
                                   int mods) {
  if (!screen->mouseButtonCallbackEvent(button, action, mods)) {
    renderer->mouse_event(button, action, mods);
  }
}

void Viewer::key_callback(GLFWwindow *window, int key, int scancode, int action,
                          int mods) {

  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_ESCAPE) {
      exit(0);
      // glfwSetWindowShouldClose( window, true );
    }
  }

  renderer->keyboard_event(key, action, mods);
  screen->keyCallbackEvent(key, scancode, action, mods);
}

} // namespace CGL
