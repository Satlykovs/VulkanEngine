#pragma once

#include <optional>
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
    void createSurface();

    void pickPhysicalDevice();

    void createLogicalDevice();

    [[nodiscard]] bool checkValidationLayerSupport() const;

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        [[nodiscard]] bool isComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) const;

    const int WIDTH = 800;
    const int HEIGHT = 600;

    GLFWwindow* window_ = nullptr;

    vk::Instance instance_;
    vk::SurfaceKHR surface_;

    vk::PhysicalDevice physicalDevice_ = nullptr;

    vk::Device device_;

    vk::Queue graphicsQueue_;
    vk::Queue presentQueue_;

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};