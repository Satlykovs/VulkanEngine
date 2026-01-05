#include "VulkanEngine.hpp"

#include <spdlog/spdlog.h>
#define GLFW_INCLUDE_VULKAN
#include <algorithm>
#include <glfw/glfw3.h>
#include <iostream>
#include <limits>
#include <set>

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

    for (auto imageView : swapChainImageViews_)
    {
        device_.destroyImageView(imageView);
    }

    if (device_)
    {
        device_.destroySwapchainKHR(swapChain_);
        device_.destroy();
    }

    if (instance_ && surface_)
    {
        instance_.destroySurfaceKHR(surface_);
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

        createSurface();

        pickPhysicalDevice();
        createLogicalDevice();

        createSwapChain();
        createImageViews();
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

        if (device.getSurfaceSupportKHR(i, surface_))
        {
            indices.presentFamily = i;
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

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    float queuePriority = 1.0F;

    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePriority);

        queueCreateInfos.push_back(queueCreateInfo);
    }

    const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo({}, queueCreateInfos, {}, deviceExtensions, &deviceFeatures);

    try
    {
        device_ = physicalDevice_.createDevice(createInfo);
        spdlog::info("Logical Device created successfully");

        graphicsQueue_ = device_.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue_ = device_.getQueue(indices.presentFamily.value(), 0);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create Logical Device: ") + err.what());
    }
}

void VulkanEngine::createSurface()
{
    VkSurfaceKHR surface;

    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface");
    }

    surface_ = surface;
    spdlog::info("Window Surface created successfully");
}

VulkanEngine::SwapChainSupportDetails VulkanEngine::querySwapChainSupport(
    const vk::PhysicalDevice& device) const
{
    SwapChainSupportDetails details;

    details.capabilities = device.getSurfaceCapabilitiesKHR(surface_);
    details.formats = device.getSurfaceFormatsKHR(surface_);
    details.presentModes = device.getSurfacePresentModesKHR(surface_);

    return details;
}

vk::SurfaceFormatKHR VulkanEngine::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR VulkanEngine::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox)
        {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VulkanEngine::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width;
    int height;
    glfwGetFramebufferSize(window_, &width, &height);

    vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);

    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);

    return actualExtent;
}

void VulkanEngine::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice_);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo({}, surface_, imageCount, surfaceFormat.format,
                                          surfaceFormat.colorSpace, extent, 1,
                                          vk::ImageUsageFlagBits::eColorAttachment);

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = nullptr;

    try
    {
        swapChain_ = device_.createSwapchainKHR(createInfo);
        spdlog::info("Swapchain created successfully");
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create Swapchain: ") + err.what());
    }

    swapChainImages_ = device_.getSwapchainImagesKHR(swapChain_);

    swapChainExtent_ = extent;
    swapChainImageFormat_ = surfaceFormat.format;
}

void VulkanEngine::createImageViews()
{
    swapChainImageViews_.resize(swapChainImages_.size());

    for (size_t i = 0; i < swapChainImages_.size(); ++i)
    {
        vk::ImageViewCreateInfo createInfo{};
        createInfo.image = swapChainImages_[i];

        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = swapChainImageFormat_;

        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;

        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        try
        {
            swapChainImageViews_[i] = device_.createImageView(createInfo);
        }
        catch (const vk::SystemError& err)
        {
            throw std::runtime_error(std::string("Failed to create Image Views: ") + err.what());
        }
    }

    spdlog::info("Image Views created successfully");
}