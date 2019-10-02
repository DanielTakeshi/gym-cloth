#include <cmath>
#include <glad/glad.h>

#include <CGL/vector3D.h>
#include <nanogui/nanogui.h>
#include <zmq.hpp>

#include "clothSimulator.h"

#include "camera.h"
#include "cloth.h"
#include "collision/plane.h"
#include "collision/sphere.h"
#include "misc/camera_info.h"

using namespace nanogui;
using namespace std;

ClothSimulator::ClothSimulator(Screen *screen) {
  this->screen = screen;

  // Initialize OpenGL buffers and shaders

  wireframeShader.initFromFiles("Wireframe", "../shaders/camera.vert",
                                "../shaders/wireframe.frag");
  normalShader.initFromFiles("Normal", "../shaders/camera.vert",
                             "../shaders/normal.frag");
  phongShader.initFromFiles("Phong", "../shaders/camera.vert",
                            "../shaders/phong.frag");

  shaders.push_back(wireframeShader);
  shaders.push_back(normalShader);
  shaders.push_back(phongShader);

  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_DEPTH_TEST);
}

ClothSimulator::~ClothSimulator() {
  for (auto shader : shaders) {
    shader.free();
  }

  if (cloth) delete cloth;
  if (cp) delete cp;
  if (collision_objects) delete collision_objects;
}

void ClothSimulator::loadCloth(Cloth *cloth) { this->cloth = cloth; }

void ClothSimulator::loadClothParameters(ClothParameters *cp) { this->cp = cp; }

void ClothSimulator::loadCollisionObjects(vector<CollisionObject *> *objects) { this->collision_objects = objects; }

/**
 * Initializes the cloth simulation and spawns a new thread to separate
 * rendering from simulation.
 */
void ClothSimulator::init() {
  // Initialize GUI
  initGUI(screen);
  screen->setSize(default_window_size);

  // Initialize camera

  CGL::Collada::CameraInfo camera_info;
  camera_info.hFov = 50;
  camera_info.vFov = 35;
  camera_info.nClip = 0.01;
  camera_info.fClip = 10000;

  // Try to intelligently figure out the camera target

  Vector3D avg_pm_position(0, 0, 0);

  for (auto &pm : cloth->point_masses) {
    avg_pm_position += pm.position / cloth->point_masses.size();
  }

  CGL::Vector3D target(avg_pm_position.x, avg_pm_position.y / 2,
                       avg_pm_position.z);
  CGL::Vector3D c_dir(0., 0., 0.);
  canonical_view_distance = max(cloth->width, cloth->height) * 0.9;
  scroll_rate = canonical_view_distance / 10;

  view_distance = canonical_view_distance * 2;
  min_view_distance = canonical_view_distance / 10.0;
  max_view_distance = canonical_view_distance * 20.0;

  // canonicalCamera is a copy used for view resets

  camera.place(target, acos(c_dir.y), atan2(c_dir.x, c_dir.z), view_distance,
               min_view_distance, max_view_distance);
  canonicalCamera.place(target, acos(c_dir.y), atan2(c_dir.x, c_dir.z),
                        view_distance, min_view_distance, max_view_distance);

  screen_w = default_window_size(0);
  screen_h = default_window_size(1);

  camera.configure(camera_info, screen_w, screen_h);
  canonicalCamera.configure(camera_info, screen_w, screen_h);

  // added by @ryanhoque: rotate camera so you can see the fold better. adjust as necessary
  camera.rotate_by(1, 0.0);   // Ryan used (2, 0.0)
  camera.move_forward(-0.5);  // Ryan used -0.5
}

bool ClothSimulator::isAlive() { return is_alive; }

void ClothSimulator::drawContents() {
  glEnable(GL_DEPTH_TEST);

  if (!is_paused) {
    vector<Vector3D> external_accelerations = {gravity};

    cloth->simulate(frames_per_sec, simulation_steps, cp, external_accelerations, collision_objects);
  }

  // Bind the active shader

  GLShader shader = shaders[activeShader];
  shader.bind();

  // Prepare the camera projection matrix

  Matrix4f model;
  model.setIdentity();

  Matrix4f view = getViewMatrix();
  Matrix4f projection = getProjectionMatrix();

  Matrix4f viewProjection = projection * view;

  shader.setUniform("model", model);
  shader.setUniform("viewProjection", viewProjection);

  switch (activeShader) {
  case WIREFRAME:
    drawWireframe(shader);
    break;
  case NORMALS:
    drawNormals(shader);
    break;
  case PHONG:
    drawPhong(shader);
    break;
  }

  for (CollisionObject *co : *collision_objects) {
    co->render(shader);
  }
}

void ClothSimulator::drawWireframe(GLShader &shader) {
  int num_structural_springs =
      2 * cloth->num_width_points * cloth->num_height_points -
      cloth->num_width_points - cloth->num_height_points;
  int num_shear_springs =
      2 * (cloth->num_width_points - 1) * (cloth->num_height_points - 1);
  int num_bending_springs = num_structural_springs - cloth->num_width_points -
                            cloth->num_height_points;

  int num_springs = cp->enable_structural_constraints * num_structural_springs +
                    cp->enable_shearing_constraints * num_shear_springs +
                    cp->enable_bending_constraints * num_bending_springs;

  MatrixXf positions(3, num_springs * 2);
  MatrixXf normals(3, num_springs * 2);

  // Draw springs as lines

  int si = 0;

  for (int i = 0; i < cloth->springs.size(); i++) {
    Spring s = cloth->springs[i];

    if ((s.spring_type == STRUCTURAL && !cp->enable_structural_constraints) ||
        (s.spring_type == SHEARING && !cp->enable_shearing_constraints) ||
        (s.spring_type == BENDING && !cp->enable_bending_constraints)) {
      continue;
    }

    Vector3D pa = s.pm_a->position;
    Vector3D pb = s.pm_b->position;

    Vector3D na = s.pm_a->normal();
    Vector3D nb = s.pm_b->normal();

    positions.col(si) << pa.x, pa.y, pa.z;
    positions.col(si + 1) << pb.x, pb.y, pb.z;

    normals.col(si) << na.x, na.y, na.z;
    normals.col(si + 1) << nb.x, nb.y, nb.z;

    si += 2;
  }

  shader.setUniform("in_color", nanogui::Color(1.0f, 1.0f, 1.0f, 1.0f));
  shader.uploadAttrib("in_position", positions);
  shader.drawArray(GL_LINES, 0, num_springs * 2);
}

void ClothSimulator::drawNormals(GLShader &shader) {
  int num_tris = cloth->clothMesh->triangles.size();

  MatrixXf positions(3, num_tris * 3);
  MatrixXf normals(3, num_tris * 3);

  for (int i = 0; i < num_tris; i++) {
    Triangle *tri = cloth->clothMesh->triangles[i];

    Vector3D p1 = tri->pm1->position;
    Vector3D p2 = tri->pm2->position;
    Vector3D p3 = tri->pm3->position;

    Vector3D n1 = tri->pm1->normal();
    Vector3D n2 = tri->pm2->normal();
    Vector3D n3 = tri->pm3->normal();

    positions.col(i * 3) << p1.x, p1.y, p1.z;
    positions.col(i * 3 + 1) << p2.x, p2.y, p2.z;
    positions.col(i * 3 + 2) << p3.x, p3.y, p3.z;

    normals.col(i * 3) << n1.x, n1.y, n1.z;
    normals.col(i * 3 + 1) << n2.x, n2.y, n2.z;
    normals.col(i * 3 + 2) << n3.x, n3.y, n3.z;
  }

  shader.uploadAttrib("in_position", positions);
  shader.uploadAttrib("in_normal", normals);

  shader.drawArray(GL_TRIANGLES, 0, num_tris * 3);
}

void ClothSimulator::drawPhong(GLShader &shader) {
  int num_tris = cloth->clothMesh->triangles.size();

  MatrixXf positions(3, num_tris * 3);
  MatrixXf normals(3, num_tris * 3);

  for (int i = 0; i < num_tris; i++) {
    Triangle *tri = cloth->clothMesh->triangles[i];

    Vector3D p1 = tri->pm1->position;
    Vector3D p2 = tri->pm2->position;
    Vector3D p3 = tri->pm3->position;

    Vector3D n1 = tri->pm1->normal();
    Vector3D n2 = tri->pm2->normal();
    Vector3D n3 = tri->pm3->normal();

    positions.col(i * 3) << p1.x, p1.y, p1.z;
    positions.col(i * 3 + 1) << p2.x, p2.y, p2.z;
    positions.col(i * 3 + 2) << p3.x, p3.y, p3.z;

    normals.col(i * 3) << n1.x, n1.y, n1.z;
    normals.col(i * 3 + 1) << n2.x, n2.y, n2.z;
    normals.col(i * 3 + 2) << n3.x, n3.y, n3.z;
  }

  Vector3D cp = camera.position();

  shader.setUniform("in_color", color);
  shader.setUniform("eye", Vector3f(cp.x, cp.y, cp.z));
  shader.setUniform("light", Vector3f(0.5, 2, 2));

  shader.uploadAttrib("in_position", positions);
  shader.uploadAttrib("in_normal", normals);

  shader.drawArray(GL_TRIANGLES, 0, num_tris * 3);
}

// ----------------------------------------------------------------------------
// CAMERA CALCULATIONS
//
// OpenGL 3.1 deprecated the fixed pipeline, so we lose a lot of useful OpenGL
// functions that have to be recreated here.
// ----------------------------------------------------------------------------

void ClothSimulator::resetCamera() { camera.copy_placement(canonicalCamera); }

Matrix4f ClothSimulator::getProjectionMatrix() {
  Matrix4f perspective;
  perspective.setZero();

  double near = camera.near_clip();
  double far = camera.far_clip();

  double theta = camera.v_fov() * M_PI / 360;
  double range = far - near;
  double invtan = 1. / tanf(theta);

  perspective(0, 0) = invtan / camera.aspect_ratio();
  perspective(1, 1) = invtan;
  perspective(2, 2) = -(near + far) / range;
  perspective(3, 2) = -1;
  perspective(2, 3) = -2 * near * far / range;
  perspective(3, 3) = 0;

  return perspective;
}

Matrix4f ClothSimulator::getViewMatrix() {
  Matrix4f lookAt;
  Matrix3f R;

  lookAt.setZero();

  // Convert CGL vectors to Eigen vectors
  // TODO: Find a better way to do this!

  CGL::Vector3D c_pos = camera.position();
  CGL::Vector3D c_udir = camera.up_dir();
  CGL::Vector3D c_target = camera.view_point();

  Vector3f eye(c_pos.x, c_pos.y, c_pos.z);
  Vector3f up(c_udir.x, c_udir.y, c_udir.z);
  Vector3f target(c_target.x, c_target.y, c_target.z);

  R.col(2) = (eye - target).normalized();
  R.col(0) = up.cross(R.col(2)).normalized();
  R.col(1) = R.col(2).cross(R.col(0));

  lookAt.topLeftCorner<3, 3>() = R.transpose();
  lookAt.topRightCorner<3, 1>() = -R.transpose() * eye;
  lookAt(3, 3) = 1.0f;

  return lookAt;
}

// ----------------------------------------------------------------------------
// EVENT HANDLING
// ----------------------------------------------------------------------------

bool ClothSimulator::cursorPosCallbackEvent(double x, double y) {
  if (left_down && !middle_down && !right_down) {
    if (ctrl_down) {
      mouseRightDragged(x, y);
    } else {
      mouseLeftDragged(x, y);
    }
  } else if (!left_down && !middle_down && right_down) {
    mouseRightDragged(x, y);
  } else if (!left_down && !middle_down && !right_down) {
    mouseMoved(x, y);
  }

  mouse_x = x;
  mouse_y = y;

  return true;
}

bool ClothSimulator::mouseButtonCallbackEvent(int button, int action,
                                              int modifiers) {
  switch (action) {
  case GLFW_PRESS:
    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      left_down = true;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      middle_down = true;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      right_down = true;
      break;
    }
    return true;

  case GLFW_RELEASE:
    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      left_down = false;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      middle_down = false;
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      right_down = false;
      break;
    }
    return true;
  }

  return false;
}

void ClothSimulator::mouseMoved(double x, double y) { y = screen_h - y; }

void ClothSimulator::mouseLeftDragged(double x, double y) {
  float dx = x - mouse_x;
  float dy = y - mouse_y;

  camera.rotate_by(-dy * (PI / screen_h), -dx * (PI / screen_w));
}

void ClothSimulator::mouseRightDragged(double x, double y) {
  camera.move_by(mouse_x - x, y - mouse_y, canonical_view_distance);
}

bool ClothSimulator::keyCallbackEvent(int key, int scancode, int action,
                                      int mods) {
  ctrl_down = (bool)(mods & GLFW_MOD_CONTROL);

  if (action == GLFW_PRESS) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
      is_alive = false;
      break;
    case 'r':
    case 'R':
      cloth->reset();
      break;
    case ' ':
      resetCamera();
      break;
    case 'p':
    case 'P':
      is_paused = !is_paused;
      break;
    case 'n':
    case 'N':
      if (is_paused) {
        is_paused = false;
        drawContents();
        is_paused = true;
      }
      break;
    }
  }

  return true;
}

bool ClothSimulator::dropCallbackEvent(int count, const char **filenames) {
  return true;
}

bool ClothSimulator::scrollCallbackEvent(double x, double y) {
  camera.move_forward(y * scroll_rate);
  return true;
}

bool ClothSimulator::resizeCallbackEvent(int width, int height) {
  screen_w = width;
  screen_h = height;

  camera.set_screen_size(screen_w, screen_h);
  return true;
}

void ClothSimulator::initGUI(Screen *screen) {
  Window *window = new Window(screen, "Settings");
  window->setPosition(Vector2i(15, 15));
  window->setLayout(new GroupLayout(15, 6, 14, 5));

  // Spring types

  new Label(window, "Spring types", "sans-bold");

  {
    Button *b = new Button(window, "structural");
    b->setFlags(Button::ToggleButton);
    b->setPushed(cp->enable_structural_constraints);
    b->setFontSize(14);
    b->setChangeCallback(
        [this](bool state) { cp->enable_structural_constraints = state; });

    b = new Button(window, "shearing");
    b->setFlags(Button::ToggleButton);
    b->setPushed(cp->enable_shearing_constraints);
    b->setFontSize(14);
    b->setChangeCallback(
        [this](bool state) { cp->enable_shearing_constraints = state; });

    b = new Button(window, "bending");
    b->setFlags(Button::ToggleButton);
    b->setPushed(cp->enable_bending_constraints);
    b->setFontSize(14);
    b->setChangeCallback(
        [this](bool state) { cp->enable_bending_constraints = state; });
  }

  // Mass-spring parameters

  new Label(window, "Parameters", "sans-bold");

  {
    Widget *panel = new Widget(window);
    GridLayout *layout =
        new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 5, 5);
    layout->setColAlignment({Alignment::Maximum, Alignment::Fill});
    layout->setSpacing(0, 10);
    panel->setLayout(layout);

    new Label(panel, "density :", "sans-bold");

    FloatBox<double> *fb = new FloatBox<double>(panel);
    fb->setEditable(true);
    fb->setFixedSize(Vector2i(100, 20));
    fb->setFontSize(14);
    fb->setValue(cp->density / 10);
    fb->setUnits("g/cm^2");
    fb->setSpinnable(true);
    fb->setCallback([this](float value) { cp->density = (double)(value * 10); });

    new Label(panel, "ks :", "sans-bold");

    fb = new FloatBox<double>(panel);
    fb->setEditable(true);
    fb->setFixedSize(Vector2i(100, 20));
    fb->setFontSize(14);
    fb->setValue(cp->ks);
    fb->setUnits("N/m");
    fb->setSpinnable(true);
    fb->setMinValue(0);
    fb->setCallback([this](float value) { cp->ks = value; });
  }

  // Simulation constants

  new Label(window, "Simulation", "sans-bold");

  {
    Widget *panel = new Widget(window);
    GridLayout *layout =
        new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 5, 5);
    layout->setColAlignment({Alignment::Maximum, Alignment::Fill});
    layout->setSpacing(0, 10);
    panel->setLayout(layout);

    new Label(panel, "frames/s :", "sans-bold");

    IntBox<int> *fsec = new IntBox<int>(panel);
    fsec->setEditable(true);
    fsec->setFixedSize(Vector2i(100, 20));
    fsec->setFontSize(14);
    fsec->setValue(frames_per_sec);
    fsec->setSpinnable(true);
    fsec->setCallback([this](int value) { frames_per_sec = value; });

    new Label(panel, "steps/frame :", "sans-bold");

    IntBox<int> *num_steps = new IntBox<int>(panel);
    num_steps->setEditable(true);
    num_steps->setFixedSize(Vector2i(100, 20));
    num_steps->setFontSize(14);
    num_steps->setValue(simulation_steps);
    num_steps->setSpinnable(true);
    num_steps->setMinValue(0);
    num_steps->setCallback([this](int value) { simulation_steps = value; });
  }

  // Damping slider and textbox

  new Label(window, "Damping", "sans-bold");

  {
    Widget *panel = new Widget(window);
    panel->setLayout(
        new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5));

    Slider *slider = new Slider(panel);
    slider->setValue(cp->damping);
    slider->setFixedWidth(105);

    TextBox *percentage = new TextBox(panel);
    percentage->setFixedWidth(75);
    percentage->setValue(to_string(cp->damping));
    percentage->setUnits("%");
    percentage->setFontSize(14);

    slider->setCallback([percentage](float value) {
      percentage->setValue(std::to_string(value));
    });
    slider->setFinalCallback([&](float value) {
      cp->damping = (double)value;
      // cout << "Final slider value: " << (int)(value * 100) << endl;
    });
  }

  // Gravity

  new Label(window, "Gravity", "sans-bold");

  {
    Widget *panel = new Widget(window);
    GridLayout *layout =
        new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 5, 5);
    layout->setColAlignment({Alignment::Maximum, Alignment::Fill});
    layout->setSpacing(0, 10);
    panel->setLayout(layout);

    new Label(panel, "x :", "sans-bold");

    FloatBox<double> *fb = new FloatBox<double>(panel);
    fb->setEditable(true);
    fb->setFixedSize(Vector2i(100, 20));
    fb->setFontSize(14);
    fb->setValue(gravity.x);
    fb->setUnits("m/s^2");
    fb->setSpinnable(true);
    fb->setCallback([this](float value) { gravity.x = value; });

    new Label(panel, "y :", "sans-bold");

    fb = new FloatBox<double>(panel);
    fb->setEditable(true);
    fb->setFixedSize(Vector2i(100, 20));
    fb->setFontSize(14);
    fb->setValue(gravity.y);
    fb->setUnits("m/s^2");
    fb->setSpinnable(true);
    fb->setCallback([this](float value) { gravity.y = value; });

    new Label(panel, "z :", "sans-bold");

    fb = new FloatBox<double>(panel);
    fb->setEditable(true);
    fb->setFixedSize(Vector2i(100, 20));
    fb->setFontSize(14);
    fb->setValue(gravity.z);
    fb->setUnits("m/s^2");
    fb->setSpinnable(true);
    fb->setCallback([this](float value) { gravity.z = value; });
  }

  // Appearance

  new Label(window, "Appearance", "sans-bold");

  {
    ComboBox *cb = new ComboBox(window, {"Wireframe", "Normals", "Shaded"});
    cb->setFontSize(14);
    cb->setSelectedIndex(2);
    activeShader = static_cast<e_shader>(2);
    cb->setCallback(
        [this, screen](int idx) { activeShader = static_cast<e_shader>(idx); });
  }

  new Label(window, "Color", "sans-bold");

  {
    const nanogui::Color white = nanogui::Color(255, 255, 255, 255);
    ColorWheel *cw = new ColorWheel(window, white);
    this->color = white;
    cw->setCallback(
        [this](const nanogui::Color &color) { this->color = color; });
  }
}
