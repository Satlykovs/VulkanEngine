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


FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
)

FetchContent_MakeAvailable(spdlog)


file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp")

add_executable(MyEngine ${SOURCES})

target_link_libraries(MyEngine PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        spdlog::spdlog

)




find_program(CLANG_FORMAT_EXE clang-format)

if (CLANG_FORMAT_EXE)
    add_custom_target(
            check-format
            COMMAND ${CLANG_FORMAT_EXE} -style=file --dry-run --Werror ${SOURCES}
            COMMENT "Checking code formatting... (will fail if needs formatting)"
            VERBATIM
    )

    add_custom_target(
            format
            COMMAND ${CLANG_FORMAT_EXE} -style=file -i ${SOURCES}
            COMMENT "Formatting all source files..."
            VERBATIM
    )
    add_dependencies(MyEngine check-format)
else ()
    message(WARNING "clang-format not found! Formatting targets disabled.")
endif ()
