# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

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
CMAKE_COMMAND = /root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/cmake

# The command to remove a file.
RM = /root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build

# Include any dependencies generated for this target.
include CMakeFiles/bootstrap_benchmark.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bootstrap_benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bootstrap_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bootstrap_benchmark.dir/flags.make

CMakeFiles/bootstrap_benchmark.dir/codegen:
.PHONY : CMakeFiles/bootstrap_benchmark.dir/codegen

CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o: CMakeFiles/bootstrap_benchmark.dir/flags.make
CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o: /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/bootstrap_benchmark.cc
CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o: CMakeFiles/bootstrap_benchmark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o -MF CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o.d -o CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o -c /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/bootstrap_benchmark.cc

CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.i"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/bootstrap_benchmark.cc > CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.i

CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.s"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/bootstrap_benchmark.cc -o CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.s

CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o: CMakeFiles/bootstrap_benchmark.dir/flags.make
CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o: /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/fast_bootstrap.cc
CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o: CMakeFiles/bootstrap_benchmark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o -MF CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o.d -o CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o -c /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/fast_bootstrap.cc

CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.i"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/fast_bootstrap.cc > CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.i

CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.s"
	/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/fast_bootstrap.cc -o CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.s

# Object files for target bootstrap_benchmark
bootstrap_benchmark_OBJECTS = \
"CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o" \
"CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o"

# External object files for target bootstrap_benchmark
bootstrap_benchmark_EXTERNAL_OBJECTS =

bootstrap_benchmark: CMakeFiles/bootstrap_benchmark.dir/bootstrap_benchmark.cc.o
bootstrap_benchmark: CMakeFiles/bootstrap_benchmark.dir/fast_bootstrap.cc.o
bootstrap_benchmark: CMakeFiles/bootstrap_benchmark.dir/build.make
bootstrap_benchmark: CMakeFiles/bootstrap_benchmark.dir/compiler_depend.ts
bootstrap_benchmark: /opt/conda/lib/libmpi.so
bootstrap_benchmark: /root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/lib/libmscclpp.so
bootstrap_benchmark: CMakeFiles/bootstrap_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bootstrap_benchmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bootstrap_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bootstrap_benchmark.dir/build: bootstrap_benchmark
.PHONY : CMakeFiles/bootstrap_benchmark.dir/build

CMakeFiles/bootstrap_benchmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bootstrap_benchmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bootstrap_benchmark.dir/clean

CMakeFiles/bootstrap_benchmark.dir/depend:
	cd /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build /root/kernel_dev/curr/ucxx-dev/rebuildCCL/bootstrap/seastar_bootstrap/build/CMakeFiles/bootstrap_benchmark.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/bootstrap_benchmark.dir/depend

