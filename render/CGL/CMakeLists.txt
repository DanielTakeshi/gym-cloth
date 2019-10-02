cmake_minimum_required(VERSION 2.8)
project(CGL)

#-------------------------------------------------------------------------------
# Build options
#-------------------------------------------------------------------------------
option(CGL_BUILD_DEBUG    "Build for debug"          OFF)
option(CGL_BUILD_SHARED   "Build shared libraries"   OFF)
option(CGL_BUILD_DOCS     "Build documentation"      OFF)
option(CGL_BUILD_TESTS    "Build tests programs"     OFF)
option(CGL_BUILD_EXAMPLES "Build examples"           OFF)

#-------------------------------------------------------------------------------
# CMake modules
#-------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CGL_SOURCE_DIR}/cmake/modules/")

#-------------------------------------------------------------------------------
# CGL paths
#-------------------------------------------------------------------------------
set(CGL_INCLUDE_DIRS "${CGL_SOURCE_DIR}/include")

#-------------------------------------------------------------------------------
# Platform-specific settings
#-------------------------------------------------------------------------------

###################
# Building on OSX #
###################
if(APPLE)

  # Clang only
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "CLANG")

    # OSX Framework dependencies
    if(NOT CGL_BUILD_SHARED)
      include_directories( "/System/Library/Frameworks" )
      find_library (COCOA_LIBRARIES Cocoa)
      find_library (IOKIT_LIBRARIES IOkit)
      find_library (COREVIDEO_LIBRARIES CoreVideo)
    endif()

    # Clang configuration
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")

      set(CLANG_CXX_FLAGS "-std=c++11 -m64")

      if(CGL_BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
      else(CGL_BUILD_DEBUG)
        set(CLANG_CXX_FLAGS "${CLANG_CXX_FLAGS} -O3")
        set(CLANG_CXX_FLAGS "${CLANG_CXX_FLAGS} -funroll-loops")
        set(CLANG_CXX_FLAGS "${CLANG_CXX_FLAGS} -Wno-narrowing")
        set(CLANG_CXX_FLAGS "${CLANG_CXX_FLAGS} -Wno-deprecated-register")
      endif(CGL_BUILD_DEBUG)

      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_CXX_FLAGS}")

    endif()

    # GCC configuration
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

      set(GCC_CXX_FLAGS "-std=gnu++11 -m64")

      if(CGL_BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
      else(CGL_BUILD_DEBUG)
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -O3")
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fopenmp")
      endif(CGL_BUILD_DEBUG)

      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS}")

    endif()

  endif()

endif(APPLE)

##################
# Build on Linux #
##################
set(LINUX UNIX AND NOT APPLE)
if(LINUX)

  # GCC only
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    set(GCC_CXX_FLAGS "-std=gnu++11 -m64")

    # X11 Dependencies
    if(NOT CGL_BUILD_SHARED)
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXi")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXxf86vm")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXinerama")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXcursor")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXfixes")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXrandr")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXext")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXrender")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lX11")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lpthread")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lxcb")
      set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lXau")
    endif()

    # Debug configuration
    if(CGL_BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
    else(CGL_BUILD_DEBUG)
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -O3")
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fopenmp")
    endif(CGL_BUILD_DEBUG)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS}")

  endif()

endif(LINUX)

####################
# Build on Windows #
####################
if(WIN32)

  if(MSVC)

    set(MSVC_CXX_FLAGS "-std=gnu++11")

    if(CGL_BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
    else(CGL_BUILD_DEBUG)
    endif(CGL_BUILD_DEBUG)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MSVC_CXX_FLAGS}")

  endif(MSVC)

  if(MINGW)

    set(MSVC_CXX_FLAGS "-std=gnu++11")

    if(CGL_BUILD_DEBUG)
        set(CMAKE_BUILD_TYPE Debug)
    else(CGL_BUILD_DEBUG)
    endif(CGL_BUILD_DEBUG)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MSVC_CXX_FLAGS}")

  endif(MINGW)

endif(WIN32)

#-------------------------------------------------------------------------------
# Find dependencies
#-------------------------------------------------------------------------------

# Required packages
find_package(Freetype REQUIRED)


#-------------------------------------------------------------------------------
# Set definitions
#-------------------------------------------------------------------------------
add_definitions(${NANOGUI_EXTRA_DEFS})

#-------------------------------------------------------------------------------
# Set include directories
#-------------------------------------------------------------------------------
include_directories(
  ${CGL_INCLUDE_DIRS}
  ${FREETYPE_INCLUDE_DIRS}
  ${NANOGUI_EXTRA_INCS}
)

#-------------------------------------------------------------------------------
# Set link directories
#-------------------------------------------------------------------------------
link_directories(
  ${FREETYPE_LIBRARY_DIRS}
)

#-------------------------------------------------------------------------------
# Add subdirectories
#-------------------------------------------------------------------------------

# CGL library source directory
add_subdirectory(src)

# CGL tests source directory
if(CGL_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# CGL exmaples source directory
if(CGL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# CGL documentation directory
if(CGL_BUILD_DOCS)
  find_package(DOXYGEN)
  if(DOXYGEN_FOUND AND CGL_BUILD_DOCS)
    add_subdirectory(docs)
  endif()
endif()

#-------------------------------------------------------------------------------
# Packing
#-------------------------------------------------------------------------------

# Install settings
set(CMAKE_INSTALL_PREFIX "${CGL_SOURCE_DIR}")
