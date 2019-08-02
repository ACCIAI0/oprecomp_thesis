# CMake generated Testfile for 
# Source directory: /home/alberto/oprecomp/flexfloat/test
# Build directory: /home/alberto/oprecomp/flexfloat/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_conversion "/home/alberto/oprecomp/flexfloat/build/test/conversion")
add_test(test_assignment "/home/alberto/oprecomp/flexfloat/build/test/assignment")
add_test(test_sanitize "/home/alberto/oprecomp/flexfloat/build/test/sanitize")
add_test(test_nearest_rounding "/home/alberto/oprecomp/flexfloat/build/test/nearest_rounding")
add_test(test_exception_flags "/home/alberto/oprecomp/flexfloat/build/test/exception_flags")
add_test(test_upward_rounding "/home/alberto/oprecomp/flexfloat/build/test/upward_rounding")
add_test(test_downward_rounding "/home/alberto/oprecomp/flexfloat/build/test/downward_rounding")
add_test(test_NanInf "/home/alberto/oprecomp/flexfloat/build/test/NanInf")
add_test(test_rel_ops "/home/alberto/oprecomp/flexfloat/build/test/rel_ops")
add_test(test_arithmetic "/home/alberto/oprecomp/flexfloat/build/test/arithmetic")
add_test(test_value_representation "/home/alberto/oprecomp/flexfloat/build/test/value_representation")
add_test(test_value_representation_half "/home/alberto/oprecomp/flexfloat/build/test/value_representation_half")
subdirs("googletest-build")
