#include <cmath>
#include <nanogui/nanogui.h>

#include "sphere_drawing.h"

#include "CGL/color.h"
#include "CGL/vector3D.h"

// Static data describing points on a sphere
#define SPHERE_NUM_LAT 40
#define SPHERE_NUM_LON 40

#define SPHERE_NUM_VERTICES ((SPHERE_NUM_LAT + 1) * (SPHERE_NUM_LON + 1))
#define SPHERE_NUM_INDICES (6 * SPHERE_NUM_LAT * SPHERE_NUM_LON)
#define SINDEX(x, y) ((x) * (SPHERE_NUM_LON + 1) + (y))
#define VERTEX_SIZE 8
#define TCOORD_OFFSET 0
#define NORMAL_OFFSET 2
#define VERTEX_OFFSET 5
#define BUMP_FACTOR 1

static unsigned int Indices[SPHERE_NUM_INDICES];
static double Vertices[VERTEX_SIZE * SPHERE_NUM_VERTICES];
static bool initialized = false;
static double vertices[3 * SPHERE_NUM_VERTICES];
static double normals[3 * SPHERE_NUM_VERTICES];

using namespace nanogui;

namespace CGL {
namespace Misc {

static void init_mesh();
static void draw_sphere(const Vector3D &p, double r);

void init_mesh() {
  for (int i = 0; i <= SPHERE_NUM_LAT; i++) {
    for (int j = 0; j <= SPHERE_NUM_LON; j++) {
      double lat = ((double)i) / SPHERE_NUM_LAT;
      double lon = ((double)j) / SPHERE_NUM_LON;
      double *vptr = &Vertices[VERTEX_SIZE * SINDEX(i, j)];

      vptr[TCOORD_OFFSET + 0] = lon;
      vptr[TCOORD_OFFSET + 1] = 1 - lat;

      lat *= M_PI;
      lon *= 2 * M_PI;
      double sinlat = sin(lat);

      vptr[NORMAL_OFFSET + 0] = vptr[VERTEX_OFFSET + 0] = sinlat * sin(lon);
      vptr[NORMAL_OFFSET + 1] = vptr[VERTEX_OFFSET + 1] = cos(lat),
                           vptr[NORMAL_OFFSET + 2] = vptr[VERTEX_OFFSET + 2] =
                               sinlat * cos(lon);
    }
  }

  for (int i = 0; i < SPHERE_NUM_LAT; i++) {
    for (int j = 0; j < SPHERE_NUM_LON; j++) {
      unsigned int *iptr = &Indices[6 * (SPHERE_NUM_LON * i + j)];

      unsigned int i00 = SINDEX(i, j);
      unsigned int i10 = SINDEX(i + 1, j);
      unsigned int i11 = SINDEX(i + 1, j + 1);
      unsigned int i01 = SINDEX(i, j + 1);

      iptr[0] = i00;
      iptr[1] = i10;
      iptr[2] = i11;
      iptr[3] = i11;
      iptr[4] = i01;
      iptr[5] = i00;
    }
  }
}

void draw_sphere(GLShader &shader, const Vector3D &p, double r) {
  if (!initialized) {
    init_mesh();
    initialized = true;
  }

  Matrix4f model;
  model << r, 0, 0, p.x, 0, r, 0, p.y, 0, 0, r, p.z, 0, 0, 0, 1;

  shader.setUniform("model", model);

  MatrixXf positions(3, SPHERE_NUM_INDICES * 3);
  MatrixXf normals(3, SPHERE_NUM_INDICES * 3);

  for (int i = 0; i < SPHERE_NUM_INDICES; i += 3) {
    double *vPtr1 = &Vertices[VERTEX_SIZE * Indices[i]];
    double *vPtr2 = &Vertices[VERTEX_SIZE * Indices[i + 1]];
    double *vPtr3 = &Vertices[VERTEX_SIZE * Indices[i + 2]];

    Vector3D p1(vPtr1[VERTEX_OFFSET], vPtr1[VERTEX_OFFSET + 1],
                vPtr1[VERTEX_OFFSET + 2]);
    Vector3D p2(vPtr2[VERTEX_OFFSET], vPtr2[VERTEX_OFFSET + 1],
                vPtr2[VERTEX_OFFSET + 2]);
    Vector3D p3(vPtr3[VERTEX_OFFSET], vPtr3[VERTEX_OFFSET + 1],
                vPtr3[VERTEX_OFFSET + 2]);

    Vector3D n1(vPtr1[NORMAL_OFFSET], vPtr1[NORMAL_OFFSET + 1],
                vPtr1[NORMAL_OFFSET + 2]);
    Vector3D n2(vPtr2[NORMAL_OFFSET], vPtr2[NORMAL_OFFSET + 1],
                vPtr2[NORMAL_OFFSET + 2]);
    Vector3D n3(vPtr3[NORMAL_OFFSET], vPtr3[NORMAL_OFFSET + 1],
                vPtr3[NORMAL_OFFSET + 2]);

    positions.col(i) << p1.x, p1.y, p1.z;
    positions.col(i + 1) << p2.x, p2.y, p2.z;
    positions.col(i + 2) << p3.x, p3.y, p3.z;

    normals.col(i) << n1.x, n1.y, n1.z;
    normals.col(i + 1) << n2.x, n2.y, n2.z;
    normals.col(i + 2) << n3.x, n3.y, n3.z;
  }

  shader.uploadAttrib("in_position", positions);
  shader.uploadAttrib("in_normal", normals);

  shader.drawArray(GL_TRIANGLES, 0, SPHERE_NUM_INDICES);

  /*
  TODO: when I use this scratch code to render the mesh using the more efficient
        glDrawElements, it works for a single frame and then fails on all
        successive frames. Any opengl experts care to figure out why?
  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnable(GL_NORMALIZE);
  for (int i = 0; i < SPHERE_NUM_VERTICES; i++) {
    normals[3 * i] = Vertices[8 * i + 2];
    normals[3 * i + 1] = Vertices[8 * i + 3];
    normals[3 * i + 2] = Vertices[8 * i + 4];
    vertices[3 * i] = Vertices[8 * i + 5];
    vertices[3 * i + 1] = Vertices[8 * i + 6];
    vertices[3 * i + 2] = Vertices[8 * i + 7];
  }

  glVertexPointer(3, GL_DOUBLE, 0, vertices);
  glNormalPointer(GL_DOUBLE, 0, normals);
  glDrawElements(GL_TRIANGLES, SPHERE_NUM_INDICES, GL_UNSIGNED_INT, Indices);
  */
}

} // namespace Misc
} // namespace CGL
