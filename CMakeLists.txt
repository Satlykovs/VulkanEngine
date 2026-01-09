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


FetchContent_Declare(
        vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
)
FetchContent_MakeAvailable(vma)

FetchContent_MakeAvailable(spdlog)


file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp")

add_executable(MyEngine ${SOURCES})

target_link_libraries(MyEngine PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        spdlog::spdlog
        GPUOpen::VulkanMemoryAllocator

)


find_program(CLANG_FORMAT_EXE clang-format)

if (CLANG_FORMAT_EXE)
    add_custom_target(
            check-format
            COMMAND ${CLANG_FORMAT_EXE} -style=file --dry-run --Werror -verbose ${SOURCES}
            COMMENT "Checking code formatting..."
            VERBATIM
    )

    add_custom_target(
            format
            COMMAND ${CLANG_FORMAT_EXE} -style=file -i -verbose ${SOURCES}
            COMMENT "Formatting all source files..."
            VERBATIM
    )

    add_dependencies(MyEngine check-format)
else ()
    message(WARNING "clang-format not found!")
endif ()


find_program(GLSLC_EXE glslc HINTS $ENV{VULKAN_SDK}/bin)

if (NOT GLSLC_EXE)
    message(FATAL_ERROR "glslc not found! Check VULKAN_SDK path.")
endif ()


file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

file(GLOB_RECURSE SHADERS "shaders/*.vert" "shaders/*.frag")


foreach (SOURCE_FILE IN LISTS SHADERS)
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME)

    set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}.spv")


    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${GLSLC_EXE} ${SOURCE_FILE} -o ${OUTPUT_FILE}
            DEPENDS ${SOURCE_FILE}
            COMMENT "Compiling shader: ${SOURCE_FILE} -> ${OUTPUT_FILE}"
    )

    list(APPEND SPV_FILES ${OUTPUT_FILE})

endforeach ()


add_custom_target(Shaders ALL DEPENDS ${SPV_FILES})

add_dependencies(MyEngine Shaders)