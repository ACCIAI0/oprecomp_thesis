cmake_minimum_required(VERSION 3.1)

include(ExternalProject)
ExternalProject_Add(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           master
    SOURCE_DIR        "/home/alberto/oprecomp/flexfloat/build/test/googletest-src"
    BINARY_DIR        "/home/alberto/oprecomp/flexfloat/build/test/googletest-build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

