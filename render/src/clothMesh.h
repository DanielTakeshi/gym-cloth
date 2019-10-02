#ifndef CLOTH_MESH_H
#define CLOTH_MESH_H

#include <vector>

#include "CGL/CGL.h"
#include "pointMass.h"

using namespace CGL;
using namespace std;

class Triangle {
public:
  Triangle(PointMass *pm1, PointMass *pm2, PointMass *pm3)
      : pm1(pm1), pm2(pm2), pm3(pm3) {}

  // Static references to constituent mesh objects
  PointMass *pm1;
  PointMass *pm2;
  PointMass *pm3;

  Halfedge *halfedge;
}; // struct Triangle

class Edge {
public:
  Halfedge *halfedge;
}; // struct Edge

class Halfedge {
public:
  Edge *edge;
  Halfedge *next;
  Halfedge *twin;
  Triangle *triangle;
  PointMass *pm;
}; // struct Halfedge

class ClothMesh {
public:
  ~ClothMesh() {}

  vector<Triangle *> triangles;
}; // struct ClothMesh

#endif // CLOTH_MESH_H
