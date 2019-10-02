#ifndef CGL_PNG_H
#define CGL_PNG_H

#include <map>
#include <vector>

namespace CGL {

  struct PNG {
    int width;
    int height;
    std::vector<unsigned char> pixels;
  }; // class PNG

  class PNGParser {
  public:
    static int load( const unsigned char* buffer, size_t size, PNG& png );
    static int load( const char* filename, PNG& png );
    static int save( const char* filename, const PNG& png );
  }; // class PNGParser

} // namespace CGL

#endif // CGL_PNG_H
