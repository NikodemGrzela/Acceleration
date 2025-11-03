cmake_minimum_required(VERSION 4.1)

set (CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(akceleracja)


# debug symbols are included for the Debug configuration.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

include_directories(C:/msys64/mingw64/include)

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(APPEND MY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c"
)

file(GLOB SHADERS_TO_COPY CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/*"
)
# --- Create the Executable Target -    --
# 
# Define the executable target. The target name will be 'my_project'.
add_executable(${CMAKE_PROJECT_NAME} ${MY_SOURCES})

target_compile_features(${CMAKE_PROJECT_NAME} PUBLIC cxx_std_23)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE 
    # Fix MinGW C library headers not exposing C11 functions:
    _WIN32_WINNT=0x0601 
    # Use the Windows 7 SDK version (0x0601) to enable modern C functions.
    
    # Keeping the other definition just in case, though it may be redundant now:
    _GLIBCXX_USE_C99_STDINT_TR1 
)

# --- Link Libraries ---

# Link the executable against the required libraries.
# CMake assumes these libraries are available in your compiler's/system's library path.
target_link_libraries(${CMAKE_PROJECT_NAME}
    opengl32
    glfw3
    gdi32
    stdc++exp
)

# Optional: Copy resources to the build directory after compilation
# This assumes your resources are needed relative to the executable for runtime.
file(COPY ${SHADERS_TO_COPY} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)