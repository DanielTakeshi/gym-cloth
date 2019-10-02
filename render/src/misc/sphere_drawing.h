#ifndef CGL_UTIL_SPHEREDRAWING_H
#define CGL_UTIL_SPHEREDRAWING_H

#include <nanogui/nanogui.h>

#include "CGL/CGL.h"

using namespace nanogui;

namespace CGL {
namespace Misc {

/**
 * Draws a sphere with the given position and radius in opengl, using the
 * current modelview/projection matrices and color/material settings.
 */
void draw_sphere(GLShader &shader, const Vector3D &p, double r);

} // namespace Misc
} // namespace CGL

#endif // CGL_UTIL_SPHEREDRAWING_H
