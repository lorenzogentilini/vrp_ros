cmake_minimum_required(VERSION 3.0.2)
project(vrp_ros)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

file(GLOB python_scripts_DIR
  "src/*.py"
)

catkin_python_setup()
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a Python executable
catkin_install_python(
  PROGRAMS ${python_scripts_DIR}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)