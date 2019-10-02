#ifndef CLOTH_H
#define CLOTH_H

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

#include "CGL/CGL.h"
#include "CGL/misc.h"
#include "clothMesh.h"
#include "collision/collisionObject.h"
#include "spring.h"

using namespace CGL;
using namespace std;

enum e_orientation { HORIZONTAL = 0, VERTICAL = 1 };

/**
 * Create `ClothParameters` and `Cloth` data structures. A bunch of stuff
 * without definitions; look at `cloth.cpp` for details. The parameters here are
 * useful for rendering. Don't worry too much about these, look at `Cloth`.
 * 
 * Note that tilde operators are destructors. Look them up:
 * https://stackoverflow.com/questions/5343437/meaning-of-tilde-symbol-in-c
 */

struct ClothParameters {
  ClothParameters() {}
  ClothParameters(bool enable_structural_constraints,
                  bool enable_shearing_constraints,
                  bool enable_bending_constraints, double damping,
                  double density, double ks)
      : enable_structural_constraints(enable_structural_constraints),
        enable_shearing_constraints(enable_shearing_constraints),
        enable_bending_constraints(enable_bending_constraints),
        damping(damping), density(density), ks(ks) {}
  ~ClothParameters() {}

  // Global simulation parameters

  bool enable_structural_constraints;
  bool enable_shearing_constraints;
  bool enable_bending_constraints;

  double damping;

  // Mass-spring parameters
  double density;
  double ks;
};


/**
 * Need spatial hashing with the cloth, requires the hash position from a given
 * Vector3D (i.e., a 3D position in whatever space we're using).
 */

struct Cloth {
  Cloth() {}
  Cloth(double width, double height, int num_width_points,
        int num_height_points, float thickness);
  ~Cloth();

  void buildGrid();

  void simulate(double frames_per_sec, double simulation_steps, ClothParameters *cp,
                vector<Vector3D> external_accelerations,
                vector<CollisionObject *> *collision_objects);

  void reset();
  void buildClothMesh();

  void build_spatial_map();
  void self_collide(PointMass &pm, double simulation_steps);
  float hash_position(Vector3D pos);

  // Cloth properties
  double width;
  double height;
  int num_width_points;
  int num_height_points;
  double thickness;
  e_orientation orientation;

  // Cloth components
  vector<PointMass> point_masses;
  vector<vector<int>> pinned;
  vector<Spring> springs;
  ClothMesh *clothMesh;
  zmq::socket_t *subscriber;

  // Spatial hashing
  unordered_map<float, vector<PointMass *> *> map;
};

#endif /* CLOTH_H */
