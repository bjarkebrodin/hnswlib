# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/bb/Research/hnswlib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/bb/Research/hnswlib

# Include any dependencies generated for this target.
include CMakeFiles/multivector_search_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/multivector_search_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/multivector_search_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multivector_search_test.dir/flags.make

CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o: CMakeFiles/multivector_search_test.dir/flags.make
CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o: tests/cpp/multivector_search_test.cpp
CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o: CMakeFiles/multivector_search_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/bb/Research/hnswlib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o -MF CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o.d -o CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o -c /Users/bb/Research/hnswlib/tests/cpp/multivector_search_test.cpp

CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/bb/Research/hnswlib/tests/cpp/multivector_search_test.cpp > CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.i

CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/bb/Research/hnswlib/tests/cpp/multivector_search_test.cpp -o CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.s

# Object files for target multivector_search_test
multivector_search_test_OBJECTS = \
"CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o"

# External object files for target multivector_search_test
multivector_search_test_EXTERNAL_OBJECTS =

multivector_search_test: CMakeFiles/multivector_search_test.dir/tests/cpp/multivector_search_test.cpp.o
multivector_search_test: CMakeFiles/multivector_search_test.dir/build.make
multivector_search_test: CMakeFiles/multivector_search_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/bb/Research/hnswlib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable multivector_search_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multivector_search_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multivector_search_test.dir/build: multivector_search_test
.PHONY : CMakeFiles/multivector_search_test.dir/build

CMakeFiles/multivector_search_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multivector_search_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multivector_search_test.dir/clean

CMakeFiles/multivector_search_test.dir/depend:
	cd /Users/bb/Research/hnswlib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib/CMakeFiles/multivector_search_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/multivector_search_test.dir/depend

