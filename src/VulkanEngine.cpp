#include "VulkanEngine.hpp"

#include <spdlog/spdlog.h>
#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#include <iostream>

void VulkanEngine::init()
{
    spdlog::info("Initializing Engine...");
    initWindow();
    initVulkan();
}

void VulkanEngine::run()
{
    spdlog::info("Starting Main Loop");
    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();
    }

    spdlog::info("Main Loop Ended");
}

void VulkanEngine::cleanup()
{
    spdlog::info("Cleaning up...");

    if (instance_)
    {
        instance_.destroy();
    }

    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
    }

    glfwTerminate();
}

void VulkanEngine::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Engine", nullptr, nullptr);
}

void VulkanEngine::initVulkan()
{
    if (enableValidationLayers && !checkValidationLayerSupport())
    {
        throw std::runtime_error("Validation layers requested, but not available");
    }

    vk::ApplicationInfo appInfo("My Vulkan Engine", VK_MAKE_VERSION(1, 0, 0), "No Engine",
                                VK_MAKE_VERSION(1, 0, 0), vk::ApiVersion13);

    uint32_t glfwExtensionCount = 0;

    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    vk::InstanceCreateInfo createInfo({}, &appInfo, {}, extensions);

    if (enableValidationLayers)
    {
        createInfo.setPEnabledLayerNames(validationLayers);
        spdlog::info("Validation Layers: ENABLED");
    }

    try
    {
        instance_ = vk::createInstance(createInfo);
        spdlog::info("Vulkan Instance created successfully");
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create instance: ") + err.what());
    }
}

[[nodiscard]] bool VulkanEngine::checkValidationLayerSupport() const
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            spdlog::error("Missing layer: {}", layerName);
            return false;
        }
    }
    return true;
}