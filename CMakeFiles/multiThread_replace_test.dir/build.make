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
include CMakeFiles/multiThread_replace_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/multiThread_replace_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/multiThread_replace_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/multiThread_replace_test.dir/flags.make

CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o: CMakeFiles/multiThread_replace_test.dir/flags.make
CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o: tests/cpp/multiThread_replace_test.cpp
CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o: CMakeFiles/multiThread_replace_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/bb/Research/hnswlib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o -MF CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o.d -o CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o -c /Users/bb/Research/hnswlib/tests/cpp/multiThread_replace_test.cpp

CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/bb/Research/hnswlib/tests/cpp/multiThread_replace_test.cpp > CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.i

CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/bb/Research/hnswlib/tests/cpp/multiThread_replace_test.cpp -o CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.s

# Object files for target multiThread_replace_test
multiThread_replace_test_OBJECTS = \
"CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o"

# External object files for target multiThread_replace_test
multiThread_replace_test_EXTERNAL_OBJECTS =

multiThread_replace_test: CMakeFiles/multiThread_replace_test.dir/tests/cpp/multiThread_replace_test.cpp.o
multiThread_replace_test: CMakeFiles/multiThread_replace_test.dir/build.make
multiThread_replace_test: CMakeFiles/multiThread_replace_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/bb/Research/hnswlib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable multiThread_replace_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multiThread_replace_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/multiThread_replace_test.dir/build: multiThread_replace_test
.PHONY : CMakeFiles/multiThread_replace_test.dir/build

CMakeFiles/multiThread_replace_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/multiThread_replace_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/multiThread_replace_test.dir/clean

CMakeFiles/multiThread_replace_test.dir/depend:
	cd /Users/bb/Research/hnswlib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib /Users/bb/Research/hnswlib/CMakeFiles/multiThread_replace_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/multiThread_replace_test.dir/depend
