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

    if (device_)
    {
        device_.destroy();
    }

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

        pickPhysicalDevice();
        createLogicalDevice();
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

void VulkanEngine::pickPhysicalDevice()
{
    std::vector<vk::PhysicalDevice> devices = instance_.enumeratePhysicalDevices();

    if (devices.empty())
    {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }
    spdlog::info("Devices found: {}", devices.size());

    for (const auto& device : devices)
    {
        vk::PhysicalDeviceProperties properties = device.getProperties();

        spdlog::info(" - Checking device: {}", properties.deviceName);

        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        {
            physicalDevice_ = device;
            spdlog::info("   -> Selected Discrete GPU!");
            break;
        }
    }

    if (!physicalDevice_)
    {
        physicalDevice_ = devices[0];
        spdlog::warn("Discrete GPU not found. Using fallback: {}",
                     physicalDevice_.getProperties().deviceName);
    }

    spdlog::info("Final GPU: {}", physicalDevice_.getProperties().deviceName);
}

VulkanEngine::QueueFamilyIndices VulkanEngine::findQueueFamilies(
    const vk::PhysicalDevice& device) const
{
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
        {
            indices.graphicsFamily = i;
        }

        if (indices.isComplete())
        {
            break;
        }
        ++i;
    }
    return indices;
}

void VulkanEngine::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, indices.graphicsFamily.value(), 1,
                                              &queuePriority);

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo({}, queueCreateInfo, {}, {}, &deviceFeatures);

    try
    {
        device_ = physicalDevice_.createDevice(createInfo);
        spdlog::info("Logical Device created successfully");

        graphicsQueue_ = device_.getQueue(indices.graphicsFamily.value(), 0);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error("Failed to create Logical Device");
    }
}