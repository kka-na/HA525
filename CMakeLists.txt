cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 OLD)
project(pip LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_AUTOMOC ON) # For meta object compiler
set(CMAKE_AUTORCC ON) # Resource files
set(CMAKE_AUTOUIC ON) # UI files

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 REQUIRED Widgets Gui ) 
ADD_DEFINITIONS(${QT_DEFINITIONS})
find_package(Qt5 REQUIRED COMPONENTS OpenGL Quick Core Gui Sql Charts)
#find_package(Qt5QuickCompiler)

find_package(PkgConfig)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

find_package (OpenCV REQUIRED PATHS ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)


include_directories(/usr/local/cuda-11.4/samples/common/inc /usr/local/cuda-11.4/include)


include_directories(src src/resource src/forms src/headers src/sources)

set(HEADERS
    src/headers/mainwindow.h
    src/headers/aboutwindow.h
    src/headers/laneDetection.h
)

set(SOURCE_pip
    src/sources/mainwindow.cpp
    src/sources/aboutwindow.cpp
    src/sources/main.cpp 
    src/sources/laneDetection.cpp
    src/sources/kernel.cu
)

qt5_wrap_cpp(HEADER_pip ${HEADERS})
qt5_wrap_ui(UI_pip src/forms/mainwindow.ui src/forms/aboutwindow.ui) 
qt5_add_resources(RESOURCES src/resource/resources.qrc)
#qtquick_compiler_add_resources(RESOURCES src/resource/resources.qrc)

add_executable(pip 
    ${SOURCE_pip} ${HEADER_pip} ${UI_pip} ${RESOURCES}
)


target_link_libraries(pip ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX
    Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Charts Qt5::OpenGL
)