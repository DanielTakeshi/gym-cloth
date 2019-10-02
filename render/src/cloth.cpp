#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <zmq.hpp>
#include <string>
#include "json.hpp"

#include "cloth.h"
#include "collision/plane.h"
#include "collision/sphere.h"

using namespace std;
using json = nlohmann::json;

Cloth::Cloth(double width, double height, int num_width_points,
             int num_height_points, float thickness) {
  this->width = width;
  this->height = height;
  this->num_width_points = num_width_points;
  this->num_height_points = num_height_points;
  this->thickness = thickness;

  buildGrid();
  buildClothMesh();
}

Cloth::~Cloth() {
  point_masses.clear();
  springs.clear();

  if (clothMesh) {
    delete clothMesh;
  }
}

double frand(double min, double max) {
  double percent = (double) rand() / RAND_MAX;
  return min + percent * (max - min) ;
}

////////////////////////////////
// PART 1: MASSES AND SPRINGS //
////////////////////////////////

void Cloth::buildGrid() {
  
  double width_step = this->width / this->num_width_points;
  double height_step = this->height / this->num_height_points;
  
  // D: testing code ...
  cout << "testing printing, width_step and then height_step:\n";
  cout << width_step << "\n";         // 0.02
  cout << height_step << "\n";        // 0.02
  cout << "width/height:\n";
  cout << width << "\n";              // 1
  cout << height << "\n";             // 1
  cout << "num width/height pts:\n";
  cout << num_width_points << "\n";   // 50
  cout << num_height_points << "\n";  // 50

  // D: add stuff to Cloth->point_masses, see `cloth.h` and also the various
  // scene files, which determine (for example) which points are pinned. Also,
  // for `emplace_back` it just adds to the end of the vector, e.g., see:
  //   http://www.cplusplus.com/reference/vector/vector/emplace_back/
  //   https://www.geeksforgeeks.org/vectoremplace_back-c-stl/
  // Remember that vectors can be appended, with amortized complexity O(1).

  // Build evenly spaced grid of masses
  for (int r = 0; r < this->num_height_points; r++) {
    for (int c = 0; c < this->num_width_points; c++) {
      // if (r == 0 && c == 0) {
      //   // special case of immediately removed point
      //   this->point_masses.emplace_back(Vector3D(0, 0, 0), true); // actually should be offset
      // }
      double x  = c * width_step;
      double yz = r * height_step;
      bool pin = false;
      // Check for pin
      for (int i = 0; i < this->pinned.size(); i++) {
        if ((pinned[i][0] == c) && (pinned[i][1] == r)) {
          pin = true;
          break;
        }
      }
      if (this->orientation == HORIZONTAL) {
        this->point_masses.emplace_back(Vector3D(x, 1.0, yz), pin);
      } else {
        this->point_masses.emplace_back(Vector3D(x, yz, frand(-0.001,0.001)), pin);
      }
    }
  }

  // D: for the springs, we don't define the forces here. See `simulate` method.
  // This is purely to state to code that we should apply these in simulation.
  // Also, it takes in two point masses, and here we use (other_pt, current_pt).
  // By our design, we don't need to worry about duplicate constraints.

  // Add springs
  for (int r = 0; r < this->num_height_points; r++) {
    for (int c = 0; c < this->num_width_points; c++) {
      PointMass *cur = &this->point_masses[r * this->num_width_points + c];
      // STRUCTURAL constraint between a point mass and the point mass to its left
      if (c >= 1) {
        this->springs.emplace_back(&this->point_masses[r * this->num_width_points + c - 1], cur, STRUCTURAL);
      }
      // STRUCTURAL constraint between a point mass and the point mass above it
      if (r >= 1) {
        this->springs.emplace_back(&this->point_masses[(r - 1) * this->num_width_points + c], cur, STRUCTURAL);
      }
      // SHEARING constraints exist between a point mass and its diagonal upper left
      if (c >= 1 && r >= 1) {
        this->springs.emplace_back(&this->point_masses[(r - 1) * this->num_width_points + c - 1], cur, SHEARING);
      }
      // SHEARING constraints exist between a point mass and its diagonal upper right.
      if ((c + 1 < this->num_width_points) && r >= 1) {
        this->springs.emplace_back(&this->point_masses[(r - 1) * this->num_width_points + c + 1], cur, SHEARING);
      }
      // BENDING constraints exist between a point mass and two away to its left
      if (c >= 2) {
        this->springs.emplace_back(&this->point_masses[r * this->num_width_points + c - 2], cur, BENDING);
      }
      // BENDING constraints exist between a point mass and two away above it.
      if (r >= 2) {
        this->springs.emplace_back(&this->point_masses[(r - 2) * this->num_width_points + c], cur, BENDING);
      }
    }
  }
}

///////////////////////////////////
// PART 2: NUMERICAL INTEGRATION //
///////////////////////////////////

void Cloth::simulate(double frames_per_sec, double simulation_steps, ClothParameters *cp,
                     vector<Vector3D> external_accelerations,
                     vector<CollisionObject *> *collision_objects) {
  // overwrite method to just update point positions to what ZeroMQ says.
  zmq::message_t buffer;
  this->subscriber->recv(&buffer);
  std::string rpl = std::string(static_cast<char*>(buffer.data()), buffer.size());
  json j = json::parse(rpl);
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    int r = std::stoi(it.key().substr(0, it.key().find(" ")));
    int c = std::stoi(it.key().substr(it.key().find(" ") + 1, it.key().length()));
    // if (r == 0 && c == 0) {
    //   // special case of immediately removed point
    //   continue;
    // }
    PointMass *cur = &this->point_masses[r * this->num_width_points + c];
    cur->position = Vector3D((float) it.value()[0], (float)it.value()[1], (float)it.value()[2]);
  }
  // std::cout << rpl << std::endl;
  // float* test = j["0 0"];
  // std::cout << std::to_string(test[0]) << std::endl;
  // for (int r = 0; r < this->num_height_points; r++) {
  //   for (int c = 0; r < this->num_width_points; c++) {
  //     std::string key = std::to_string(r) + " " + std::to_string(c);
  //     auto value = *j.find(key);
  //     PointMass *cur = &this->point_masses[r * this->num_width_points + c];
  //     cur->position = Vector3D(value[0], value[1], value[2]);
  //   }
  // }

  // double mass = width * height * cp->density / num_width_points / num_height_points;
  // double delta_t = 1.0f / frames_per_sec / simulation_steps;

  // // D: ah, each iteration we reset forces to 0, as we do in Brijen's code.
  // // Also, this is external stuff, basically gravity (as the HW page states),
  // // but could include other stuff like wind if we decided to model it. Do this
  // // before all the spring constraints, which we further add to pm.forces.
  // // Update: I just checked, after running
  // //     ./clothsim -f ../scene/selfCollision.json
  // // I get (0, -9.8, 0) for all the `accl` stuff, so by default only gravity.
  // // Multiply by mass because, well, F=ma.

  // // Compute total force acting on each point mass.
  // for (PointMass &pm : this->point_masses) {
  //   pm.forces = Vector3D(0, 0, 0);
  //   for (Vector3D accl : external_accelerations) {
  //     // cout << "mass: " << mass << " accl: " << accl << "\n";
  //     pm.forces += mass * accl;
  //   }
  // }

  // for (Spring &s : this->springs) {
  //   if ((s.spring_type == STRUCTURAL && !cp->enable_structural_constraints) ||
  //       (s.spring_type == SHEARING   && !cp->enable_shearing_constraints)   ||
  //       (s.spring_type == BENDING    && !cp->enable_bending_constraints)) 
  //     continue;

  //   double KS_CORRECTION_CONSTANT = 1.0;
  //   if (s.spring_type == BENDING) {
  //     KS_CORRECTION_CONSTANT = 0.2; 
  //   }

  //   // D: Hooke's law, F_s from homework page. Aply equal but opposite forces.
  //   // Don't forget, force_mg is a FLOAT, but we need a vector for the force.
  //   // The force vector is the vector pointing from one point mass to the other
  //   // with magnitude equal to ||F_s||. That's why we do (pb - pa).unit(). Note
  //   // the direction as well, I assume pa - pb won't work. This might also be
  //   // used for the collision-collision constraints.

  //   Vector3D pa = s.pm_a->position;
  //   Vector3D pb = s.pm_b->position;
  //   double force_mg = cp->ks * KS_CORRECTION_CONSTANT * ((pb - pa).norm() - s.rest_length);
  //   Vector3D force_on_a = force_mg * (pb - pa).unit();
  //   s.pm_a->forces += force_on_a;
  //   s.pm_b->forces += -force_on_a;
  // }

  // // D: They divide damping by 100 only because their damping is a percentage,
  // // we assume we already divided by 100.
  // // D: Verlet if not pinned! We do += so that explains the first x_t term. And
  // // as expected the acceleration is from pm.forces (divided by mass). Do we
  // // handle this in Brijen's simulator? I think we assume mass is 1, which is
  // // fine, I suppose ... I know we correctly get delta^2.

  // // Use Verlet integration to compute new point mass positions
  // for (PointMass &pm : this->point_masses) {
  //   if (pm.pinned) continue;
  //   Vector3D cur_p = Vector3D(pm.position);
  //   pm.position += (1.0 - cp->damping / 100.0) * (pm.position - pm.last_position);
  //   pm.position += (pm.forces / mass) * pow(delta_t, 2);
  //   pm.last_position = cur_p;
  // }

  // // This won't do anything until you complete Part 4.
  // build_spatial_map();
  // for (PointMass &pm : point_masses) {
  //   self_collide(pm, simulation_steps);
  // }

  // // This won't do anything until you complete Part 3.
  // for (PointMass &pm : point_masses) {
  //   for (CollisionObject *co : *collision_objects) {
  //     co->collide(pm);
  //   }
  // }

  // // Constrain the changes to be such that the spring does not change
  // // in length more than 10% per timestep [Provot 1995].

  // for (Spring &s : this->springs) {
  //   Vector3D dir_ba = (s.pm_a->position - s.pm_b->position).unit();
  //   double cur_length  = (s.pm_a->position - s.pm_b->position).norm();

  //   if (cur_length > (s.rest_length * 1.1)) {
  //     double extra_diff = cur_length - s.rest_length * 1.1;
  //     if (s.pm_a->pinned && s.pm_b->pinned) {
  //       continue;
  //     } else if (s.pm_a->pinned) {
  //       s.pm_b->position += dir_ba.unit() * extra_diff;
  //     } else if (s.pm_b->pinned) {
  //       s.pm_a->position += (-dir_ba).unit() * extra_diff;
  //     } else {
  //       s.pm_a->position += (-dir_ba).unit() * extra_diff * 0.5;
  //       s.pm_b->position += dir_ba.unit() * extra_diff * 0.5;
  //     }
  //   }
  // }  
}

/////////////////////////////////////////
// PART 4: SELF COLLISIONS (3 methods) //
/////////////////////////////////////////

void Cloth::build_spatial_map() {
  for (const auto &entry : map) {
    delete(entry.second);
  }
  map.clear();

  // Build a spatial map out of all of the point masses.
  for (PointMass &pm : point_masses) {
    float hash = hash_position(pm.position);
    if (!this->map.count(hash)) {
      this->map[hash] = new vector<PointMass *>();
    }
    this->map[hash]->push_back(&pm);
  }
}

void Cloth::self_collide(PointMass &pm, double simulation_steps) {
  // D: first, get hash and see if it is contained in the map via 'count':
  // http://www.cplusplus.com/reference/unordered_map/unordered_map/count/
  // Then, we create a new 3D vector for total correction, and iterate through
  // all other points in the same hash.  If the other point is actually itself,
  // don't self-collide!
  
  // Otherwise, compute the distance between current point (pm) and the other
  // point (candidate). The `n` here is to average over all the points that were
  // close enough, i.e., distance less than 2*thickness, and we also divide by
  // simulation steps, ahh ... might need to consider that, maybe similar to
  // `physics_accuracy` in Brijen's code?
  
  // Note that we only add corrections for the CURRENT point `pm`, NOT any of
  // the other candidates! So, if a candidate point is close enough to `pm`,
  // then we will eventually get to that candidate point later when it's the
  // candidate's turn to play the role of `pm`. (Hmm ... but wait, we update
  // pm.position in real time, that would change the candidates, so might exceed
  // the thickness threshold ... right?)

  // The `correction` is pm-candidate, so it points from candidate to pm, thus
  // when we add it to pm.position, we make pm move DIRECTLY AWAY from the
  // candidate vector. The magnitude is the unit vector multiplied by the
  // 2*thick-distance, basically this means the lower the distance, the more
  // undesirable so we increase the correction force. Got it!

  // Handle self-collision for a given point mass.
  float hash = hash_position(pm.position);
  if (this->map.count(hash)) {
    Vector3D total_correction = Vector3D();
    int n = 0;
    for (PointMass *candidate : *this->map.at(hash)) {
      if (&pm == candidate) continue;
      double distance = (pm.position - candidate->position).norm();
      if (distance <= 2 * this->thickness) {
        Vector3D correction = (pm.position - candidate->position).unit() * (2 * this->thickness - distance);
        total_correction += correction;
        n++;
      }
    }
    if (n != 0) {
      pm.position += total_correction / n / simulation_steps;
    }
  }
}

float Cloth::hash_position(Vector3D pos) {
  // Hash a 3D position into a unique float identifier that represents
  // membership in some uniquely identified 3D box volume.
  double w = 3 * this->width / this->num_width_points;
  double h = 3 * this->height / this->num_height_points;
  double t = max(w, h);
  float hash = pow(31, 2) * floor(pos.x / w) + 31 * floor(pos.y / h) + floor(pos.z / t);

  // D: debugging.
  // I think Michael did this: have w, h, t as the box sizes. Then, we do
  // floor(x/w) and similarly for the others. That maps it to the unique
  // coordinates, which makes sense -- that has to be the first step. The last
  // part is to figure out how to convert the (x',y',z'), where x',y',z' are
  // integers, into a unique float that differs from the other possible integer
  // coordinate combinations. This must be O(1) and he multiplies x by a large
  // number, then multiplies y, and z gives the final offset, good. I think this
  // makes sense, as long as there are no more than 31 unique boxes in a
  // coordinate direction -- o/w at least the z would overlap (e.g., given fixed
  // x,y, if there are more than 31 boxes in the z direction this would cause
  // the post-31 boxes to overlap with some other (x,y) combo?

  // cout << "\nhash_position:\n";
  // cout << "w, h, t: " << w << ", " << h << ", " << t << "\n"; // 0.02
  // cout << "pos: " << pos << "\n";
  // cout << "floor(pos.x / w): " << floor(pos.x / w) << "\n";
  // cout << "floor(pos.y / h): " << floor(pos.y / h) << "\n";
  // cout << "floor(pos.z / t): " << floor(pos.z / t) << "\n";
  // cout << "hash: " << hash << "\n";

  return hash;
}

///////////////////////////////////////////////////////
/// YOU DO NOT NEED TO REFER TO ANY CODE BELOW THIS ///
///////////////////////////////////////////////////////

void Cloth::reset() {
  PointMass *pm = &point_masses[0];
  for (int i = 0; i < point_masses.size(); i++) {
    pm->position = pm->start_position;
    pm->last_position = pm->start_position;
    pm++;
  }
}

void Cloth::buildClothMesh() {
  if (point_masses.size() == 0) return;

  ClothMesh *clothMesh = new ClothMesh();
  vector<Triangle *> triangles;

  // Create vector of triangles
  for (int y = 0; y < num_height_points - 1; y++) {
    for (int x = 0; x < num_width_points - 1; x++) {
      PointMass *pm = &point_masses[y * num_width_points + x];
      // Both triangles defined by vertices in counter-clockwise orientation
      triangles.push_back(new Triangle(pm, pm + num_width_points, pm + 1));
      triangles.push_back(new Triangle(pm + 1, pm + num_width_points,
                                       pm + num_width_points + 1));
    }
  }

  // For each triangle in row-order, create 3 edges and 3 internal halfedges
  for (int i = 0; i < triangles.size(); i++) {
    Triangle *t = triangles[i];

    // Allocate new halfedges on heap
    Halfedge *h1 = new Halfedge();
    Halfedge *h2 = new Halfedge();
    Halfedge *h3 = new Halfedge();

    // Allocate new edges on heap
    Edge *e1 = new Edge();
    Edge *e2 = new Edge();
    Edge *e3 = new Edge();

    // Assign a halfedge pointer to the triangle
    t->halfedge = h1;

    // Assign halfedge pointers to point masses
    t->pm1->halfedge = h1;
    t->pm2->halfedge = h2;
    t->pm3->halfedge = h3;

    // Update all halfedge pointers
    h1->edge = e1;
    h1->next = h2;
    h1->pm = t->pm1;
    h1->triangle = t;

    h2->edge = e2;
    h2->next = h3;
    h2->pm = t->pm2;
    h2->triangle = t;

    h3->edge = e3;
    h3->next = h1;
    h3->pm = t->pm3;
    h3->triangle = t;
  }

  // Go back through the cloth mesh and link triangles together using halfedge
  // twin pointers

  // Convenient variables for math
  int num_height_tris = (num_height_points - 1) * 2;
  int num_width_tris = (num_width_points - 1) * 2;

  bool topLeft = true;
  for (int i = 0; i < triangles.size(); i++) {
    Triangle *t = triangles[i];

    if (topLeft) {
      // Get left triangle, if it exists
      if (i % num_width_tris != 0) { // Not a left-most triangle
        Triangle *temp = triangles[i - 1];
        t->pm1->halfedge->twin = temp->pm3->halfedge;
      } else {
        t->pm1->halfedge->twin = nullptr;
      }

      // Get triangle above, if it exists
      if (i >= num_width_tris) { // Not a top-most triangle
        Triangle *temp = triangles[i - num_width_tris + 1];
        t->pm3->halfedge->twin = temp->pm2->halfedge;
      } else {
        t->pm3->halfedge->twin = nullptr;
      }

      // Get triangle to bottom right; guaranteed to exist
      Triangle *temp = triangles[i + 1];
      t->pm2->halfedge->twin = temp->pm1->halfedge;
    } else {
      // Get right triangle, if it exists
      if (i % num_width_tris != num_width_tris - 1) { // Not a right-most triangle
        Triangle *temp = triangles[i + 1];
        t->pm3->halfedge->twin = temp->pm1->halfedge;
      } else {
        t->pm3->halfedge->twin = nullptr;
      }

      // Get triangle below, if it exists
      if (i + num_width_tris - 1 < 1.0f * num_width_tris * num_height_tris / 2.0f) { // Not a bottom-most triangle
        Triangle *temp = triangles[i + num_width_tris - 1];
        t->pm2->halfedge->twin = temp->pm3->halfedge;
      } else {
        t->pm2->halfedge->twin = nullptr;
      }

      // Get triangle to top left; guaranteed to exist
      Triangle *temp = triangles[i - 1];
      t->pm1->halfedge->twin = temp->pm2->halfedge;
    }

    topLeft = !topLeft;
  }

  clothMesh->triangles = triangles;
  this->clothMesh = clothMesh;
}
