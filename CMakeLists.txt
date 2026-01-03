cmake_minimum_required(VERSION 3.25)
project(MyVulkanEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
message(STATUS "Vulkan SDK found: ${Vulkan_FOUND}")

include(FetchContent)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
)
FetchContent_MakeAvailable(glm)


add_executable(MyEngine main.cpp)

target_link_libraries(MyEngine PRIVATE
        Vulkan::Vulkan
        glfw
        glm
)