# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/easy_webrtc_server

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/easy_webrtc_server

# Include any dependencies generated for this target.
include CMakeFiles/publish.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/publish.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/publish.dir/flags.make

CMakeFiles/publish.dir/example/publish.cc.o: CMakeFiles/publish.dir/flags.make
CMakeFiles/publish.dir/example/publish.cc.o: example/publish.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/easy_webrtc_server/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/publish.dir/example/publish.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/publish.dir/example/publish.cc.o -c /home/easy_webrtc_server/example/publish.cc

CMakeFiles/publish.dir/example/publish.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/publish.dir/example/publish.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/easy_webrtc_server/example/publish.cc > CMakeFiles/publish.dir/example/publish.cc.i

CMakeFiles/publish.dir/example/publish.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/publish.dir/example/publish.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/easy_webrtc_server/example/publish.cc -o CMakeFiles/publish.dir/example/publish.cc.s

# Object files for target publish
publish_OBJECTS = \
"CMakeFiles/publish.dir/example/publish.cc.o"

# External object files for target publish
publish_EXTERNAL_OBJECTS =

publish: CMakeFiles/publish.dir/example/publish.cc.o
publish: CMakeFiles/publish.dir/build.make
publish: librtc.a
publish: /home/muduo/build/lib/libmuduo_http.a
publish: /home/muduo/build/lib/libmuduo_net.a
publish: /home/muduo/build/lib/libmuduo_base.a
publish: CMakeFiles/publish.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/easy_webrtc_server/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable publish"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/publish.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/publish.dir/build: publish

.PHONY : CMakeFiles/publish.dir/build

CMakeFiles/publish.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/publish.dir/cmake_clean.cmake
.PHONY : CMakeFiles/publish.dir/clean

CMakeFiles/publish.dir/depend:
	cd /home/easy_webrtc_server && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/easy_webrtc_server /home/easy_webrtc_server /home/easy_webrtc_server /home/easy_webrtc_server /home/easy_webrtc_server/CMakeFiles/publish.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/publish.dir/depend

