#include <nanogui/nanogui.h>

#include "../clothMesh.h"
#include "../misc/sphere_drawing.h"
#include "sphere.h"

using namespace nanogui;
using namespace CGL;

void Sphere::collide(PointMass &pm) {
  // D: First, detect if position from pt and sphere center is less than radius.
  // that means the point is in the sphere and must be 'pushed outside'. Hence,
  // `op.norm()`.  Sphere center is: this->origin.

  // WAIT! Don't get confused ... if collision, pm.position is INSIDE the
  // sphere, while pm.last_position is OUTSIDE! The 'correction' vector points
  // from the last position (outside the sphere) TOWARDS the tangent point!!
  // THAT is why we scale it down, so that we don't actually hit the tangent but
  // stop a little further outside.

  // Handle collisions with spheres.
  Vector3D op = pm.position - this->origin;
  if (op.norm() < this->radius) {
    Vector3D tangent = this->origin + op.unit() * this->radius;
    Vector3D correction = tangent - pm.last_position;
    pm.position = pm.last_position + correction * (1 - this->friction);
  }
}

void Sphere::render(GLShader &shader) {
  // We decrease the radius here so flat triangles don't behave strangely
  // and intersect with the sphere when rendered
  Misc::draw_sphere(shader, origin, radius * 0.92);
}
