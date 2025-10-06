cmake_minimum_required(VERSION 3.5.0)
project(IntelStream VERSION 0.1.0)



# set( LRS_TARGET realsense2 )
# project( ${LRS_TARGET} LANGUAGES CXX C )

link_directories(${CMAKE_SOURCE_DIR}/lib)

file(GLOB_RECURSE SOURCE_FILES 
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADER_FILES 
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp)


# add_library(${LRS_TARGET})
include_directories(${CMAKE_BINARY_DIR}/src)

set(MNN_DIR "/home/rick/Misc/MobileNetV2/MNN")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
# set_target_properties(${LRS_TARGET} PROPERTIES VERSION ${REALSENSE_VERSION_STRING} SOVERSION "${REALSENSE_VERSION_MAJOR}.${REALSENSE_VERSION_MINOR}")

# — Correct root for your TF source (contains the 'tensorflow' subfolder) —
set(TFLITE_ROOT    "/home/rick/tensorflow")
set(TFLITE_LIB     "${TFLITE_ROOT}/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a")
set(FLATBUFFERS_INC "${TFLITE_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include")


# Find required packages
# Find OpenAL
find_package(OpenAL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(libfreenect REQUIRED)
find_package(PkgConfig REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenMP REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MNN_DIR}/include)
include_directories(${LIBUSB_INCLUDE_DIRS} ${FREENECT_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(${MNN_DIR}/lib)
# Add your executable or project files
add_executable(IntelStream ${SOURCE_FILES})

# — Tell CMake where to find the TF Lite headers —
target_include_directories(IntelStream
  PRIVATE
    "${TFLITE_ROOT}"      # so #include "tensorflow/lite/interpreter.h" works :contentReference[oaicite:9]{index=9}
    "${FLATBUFFERS_INC}"  # for flatbuffers headers :contentReference[oaicite:10]{index=10}
)

# — Import the TF Lite static library —
add_library(tensorflow-lite STATIC IMPORTED)
set_target_properties(tensorflow-lite PROPERTIES
  IMPORTED_LOCATION "${TFLITE_LIB}"
)

# — Link all dependencies in one keyword-style call —
find_package(Threads REQUIRED)
# target_link_libraries(IntelStream
#   PRIVATE
#     tensorflow-lite        # your built TF Lite .a :contentReference[oaicite:11]{index=11}
#     Threads::Threads       # for pthread support :contentReference[oaicite:12]{index=12}
#     m                      # math library :contentReference[oaicite:13]{index=13}
#     ${OpenCV_LIBS}         # if you still use OpenCV :contentReference[oaicite:14]{index=14}
#     # …add any other needed libs here…
# )


# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(IntelStream OpenMP::OpenMP_CXX)
endif()
target_link_libraries(IntelStream ${FREENECT_LIBRARIES} ${OpenCV_LIBS} ${LIBUSB_LIBRARIES})
link_libraries(${GLEW_LIBRARIES})
# Vincular librerías
target_link_libraries(IntelStream MNN)
target_link_libraries(IntelStream tensorflow-lite)
target_link_libraries(IntelStream OpenGL::OpenGL)
target_link_libraries(IntelStream GLEW::GLEW)
target_link_libraries(IntelStream Eigen3::Eigen)
# target_link_libraries( ${LRS_TARGET} PUBLIC rsutils )
target_link_libraries(IntelStream GLUT::GLUT)
target_link_libraries(IntelStream ${LIBUSB_LIBRARIES})
# Link OpenAL to your executable
target_link_libraries(IntelStream OpenAL::OpenAL)
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})

# Link the math library
target_link_libraries(IntelStream m)

# Set C++ standard
set_property(TARGET IntelStream PROPERTY CXX_STANDARD 17)
