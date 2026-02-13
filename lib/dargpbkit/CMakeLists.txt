cmake_minimum_required(VERSION 3.16)
project(dargpbkit LANGUAGES C)

set(PBGL_VERSION 0.1)
set(PBGL_GL_VERSION 1.2)

add_library(dargpbkit
  src/ogl/array.c
  src/ogl/error.c
  src/ogl/fog.c
  src/ogl/immediate.c
  src/ogl/info.c
  src/ogl/light.c
  src/ogl/matrix.c
  src/ogl/memory.c
  src/ogl/misc.c
  src/ogl/pbgl.c
  src/ogl/state.c
  src/ogl/swizzle.c
  src/ogl/texenv.c
  src/ogl/texgen.c
  src/ogl/texture.c
  src/ogl/types.c
  src/pbkit.c
  src/pbkit_dma.c
  src/pbkit_draw.c
  src/pbkit_gamma.c
  src/pbkit_print.c
  src/pbkit_pushbuffer.c
)

add_definitions(
  -DXBOX
  -DPBGL_DEBUG
  -DPBGL_VERSION=0.1
  -DPBGL_GL_VERSION=1.2
)

target_include_directories(dargpbkit PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
