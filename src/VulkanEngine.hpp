#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

struct GLFWwindow;

class VulkanEngine
{
public:
    void init();
    void run();
    void cleanup();

private:
    void initWindow();
    void initVulkan();

    [[nodiscard]] bool checkValidationLayerSupport() const;

    const int WIDTH = 800;
    const int HEIGHT = 600;

    GLFWwindow* window_ = nullptr;

    vk::Instance instance_;

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};
