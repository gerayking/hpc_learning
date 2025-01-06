# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/autodl-tmp/hpc_learning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/autodl-tmp/hpc_learning/build

# Include any dependencies generated for this target.
include CMakeFiles/vector_add.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/vector_add.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/vector_add.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vector_add.dir/flags.make

CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o: CMakeFiles/vector_add.dir/flags.make
CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o: ../src/ops/cuda/vector_add.cu
CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o: CMakeFiles/vector_add.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/autodl-tmp/hpc_learning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o -MF CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o.d -x cu -c /root/autodl-tmp/hpc_learning/src/ops/cuda/vector_add.cu -o CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o

CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target vector_add
vector_add_OBJECTS = \
"CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o"

# External object files for target vector_add
vector_add_EXTERNAL_OBJECTS =

libvector_add.a: CMakeFiles/vector_add.dir/src/ops/cuda/vector_add.cu.o
libvector_add.a: CMakeFiles/vector_add.dir/build.make
libvector_add.a: CMakeFiles/vector_add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/autodl-tmp/hpc_learning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libvector_add.a"
	$(CMAKE_COMMAND) -P CMakeFiles/vector_add.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vector_add.dir/build: libvector_add.a
.PHONY : CMakeFiles/vector_add.dir/build

CMakeFiles/vector_add.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vector_add.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vector_add.dir/clean

CMakeFiles/vector_add.dir/depend:
	cd /root/autodl-tmp/hpc_learning/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/autodl-tmp/hpc_learning /root/autodl-tmp/hpc_learning /root/autodl-tmp/hpc_learning/build /root/autodl-tmp/hpc_learning/build /root/autodl-tmp/hpc_learning/build/CMakeFiles/vector_add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vector_add.dir/depend
