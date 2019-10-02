#include "clothMesh.h"
#include <iostream>

using namespace CGL;
using namespace std;

Vector3D PointMass::normal() {
  Vector3D n(0, 0, 0);

  Halfedge *start = halfedge;
  Halfedge *iter = start;

  // Loop CCW
  do {
    n = n + cross(iter->next->pm->position - position, iter->next->next->pm->position - position);
    if (iter->next->next->twin) {
      iter = iter->next->next->twin;
    } else {
      break;
    }
  } while (iter != start);

  if (iter != start ) {
    // Terminated early from last loop; loop CW now
    start = halfedge;
    iter = start;
    if (iter->twin) {
      do {
        n = n + cross(iter->twin->next->next->pm->position - position, iter->twin->pm->position - position);
        if (iter->twin->next->twin) {
          iter = iter->twin->next;
        } else {
          break;
        }
      } while (iter != start);
    }
  }

  return n.unit();
}
