#include "spectrum.h"

#include <stdio.h>
#include <string.h>
#include <ostream>
#include <sstream>

using namespace std;

namespace CGL {

std::ostream& operator<<(std::ostream& os, const Spectrum& c) {
  os << "(r=" << c.r;
  os << " g=" << c.g;
  os << " b=" << c.b;
  os << ")";
  return os;
}

}  // namespace CGL
