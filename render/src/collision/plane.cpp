#include "iostream"
#include <nanogui/nanogui.h>

#include "../clothMesh.h"
#include "../clothSimulator.h"
#include "plane.h"

using namespace std;
using namespace CGL;

#define SURFACE_OFFSET 0.0001

void Plane::collide(PointMass &pm) {
  // D: See https://en.wikipedia.org/wiki/Plane_(geometry) for review.

  // A plane in 3D can be characterized by a point and a normal vector. See
  // `plane.h` for those declarations, both of which are type Vector3D.  Dot
  // products make sense! If dot(v1,normal) > 0 then they are pointing to the
  // same 'side' of the plane! But, I wonder, that assumes we have the normal
  // vector in a particular direction? The normal vector can't be on the other
  // side ... well we only have one plane so it's a bit easy to test by
  // inspecting the normal vector, right?

  // I think what Michael did is figure out the projection of the vector onto
  // the nearest point on the plane ... but maybe we have to take into account
  // the direction/angle? It is not generally the nearest point, right? I'll
  // go over this later.

  // Handle collisions with planes.
  Vector3D v1 = pm.last_position - this->point;
  Vector3D v2 = pm.position - this->point;
  if (dot(v1, this->normal) >= 0 && dot(v2, this->normal) <= 0) {
    double t = dot(this->point - pm.last_position, this->normal) / dot(-this->normal, this->normal);
    Vector3D tangent = pm.last_position + t * (-this->normal);
    Vector3D goal = tangent + SURFACE_OFFSET * this->normal;
    Vector3D correction = goal - pm.last_position;
    pm.position = pm.last_position + correction * (1 - this->friction);
  }
}


void Plane::render(GLShader &shader) {
  nanogui::Color color(3, 3, 91, 255);

  Vector3f sPoint(point.x, point.y, point.z);
  Vector3f sNormal(normal.x, normal.y, normal.z);
  // Hardcoded for demo
  //Vector3f sParallel(normal.y - normal.z, normal.z - normal.x,
  //                   normal.x - normal.y);
  Vector3f sParallel(normal.y - normal.z, 0,
                     normal.x - normal.y);
  sParallel.normalize();
  Vector3f sCross = sNormal.cross(sParallel);

  MatrixXf positions(3, 4);
  MatrixXf normals(3, 4);

  float size = 0.5; // modify this coefficient to change the size of the plane
  positions.col(0) << sPoint + size * (sCross + sParallel);
  positions.col(1) << sPoint + size * (sCross - sParallel);
  positions.col(2) << sPoint + size * (-sCross + sParallel);
  positions.col(3) << sPoint + size * (-sCross - sParallel);

  normals.col(0) << sNormal;
  normals.col(1) << sNormal;
  normals.col(2) << sNormal;
  normals.col(3) << sNormal;

  if (shader.uniform("in_color", false) != -1) {
    shader.setUniform("in_color", color);
  }
  shader.uploadAttrib("in_position", positions);
  shader.uploadAttrib("in_normal", normals);

  shader.drawArray(GL_TRIANGLE_STRIP, 0, 4);
}
