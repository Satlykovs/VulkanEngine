#include "VulkanEngine.hpp"

#include <spdlog/spdlog.h>
#define GLFW_INCLUDE_VULKAN
#include <algorithm>
#include <fstream>
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
        drawFrame();
    }

    device_.waitIdle();

    spdlog::info("Main Loop Ended");
}

void VulkanEngine::cleanup()
{
    spdlog::info("Cleaning up...");

    if (device_)
    {
        device_.waitIdle();

        for (int i = 0; i < swapChainImages_.size(); ++i)
        {
            device_.destroySemaphore(renderFinishedSemaphores_[i]);
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            device_.destroySemaphore(imageAvailableSemaphores_[i]);
            device_.destroyFence(inFlightFences_[i]);
        }
        device_.destroyCommandPool(commandPool_);

        device_.destroyPipeline(graphicsPipeline_);
        device_.destroyPipelineLayout(pipelineLayout_);

        for (auto imageView : swapChainImageViews_)
        {
            device_.destroyImageView(imageView);
        }

        device_.destroySwapchainKHR(swapChain_);

        vmaDestroyAllocator(allocator_);
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

        createAllocator();

        createSwapChain();
        createImageViews();

        createGraphicsPipeline();

        initSyncObjects();
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

    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

    vk::PhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.pNext = &dynamicRenderingFeatures;

    vk::DeviceCreateInfo createInfo({}, queueCreateInfos, {}, deviceExtensions, nullptr);

    createInfo.pNext = &deviceFeatures2;

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

std::vector<char> VulkanEngine::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());

    std::vector<char> buffer(fileSize);
    file.seekg(0);

    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

vk::ShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code)
{
    vk::ShaderModuleCreateInfo createInfo({}, code.size(),
                                          reinterpret_cast<const uint32_t*>(code.data()));

    try
    {
        return device_.createShaderModule(createInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create Shader Module: ") + err.what());
    }
}

void VulkanEngine::createGraphicsPipeline()
{
    auto vertShaderCode = readFile("shaders/shader.vert.spv");
    auto fragShaderCode = readFile("shaders/shader.frag.spv");

    vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex,
                                                          vertShaderModule, "main");

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment,
                                                          fragShaderModule, "main");

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 0, nullptr, 0, nullptr);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList,
                                                           VK_FALSE);

    vk::Viewport viewport(0.0F, 0.0F, static_cast<float>(swapChainExtent_.width),
                          static_cast<float>(swapChainExtent_.height), 0.0F, 1.0F);

    vk::Rect2D scissor({0, 0}, swapChainExtent_);

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
        vk::FrontFace::eClockwise, VK_FALSE, 0.0F, 0.0F, 0.0F, 1.0F);

    vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1,
                                                        &colorBlendAttachment);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 0, nullptr, 0, nullptr);

    try
    {
        pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create Pipeline Layout: ") + err.what());
    }

    vk::PipelineRenderingCreateInfo renderingInfo({}, 1, &swapChainImageFormat_);
    renderingInfo.depthAttachmentFormat = vk::Format::eUndefined;
    renderingInfo.stencilAttachmentFormat = vk::Format::eUndefined;

    // clang-format off
    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {},
        2, shaderStages,
        &vertexInputInfo,
        &inputAssembly,
        nullptr,
        &viewportState,
        &rasterizer,
        &multisampling,
        nullptr,
        &colorBlending,
        nullptr,
        pipelineLayout_,
        nullptr
    );
    // clang-format on

    pipelineInfo.pNext = &renderingInfo;

    auto result = device_.createGraphicsPipeline(nullptr, pipelineInfo);
    if (result.result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to create Graphics Pipeline");
    }

    graphicsPipeline_ = result.value;

    spdlog::info("Graphics Pipeline created successfully");

    device_.destroyShaderModule(vertShaderModule);
    device_.destroyShaderModule(fragShaderModule);
}

void VulkanEngine::initSyncObjects()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_);

    vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       queueFamilyIndices.graphicsFamily.value());

    commandPool_ = device_.createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo(commandPool_, vk::CommandBufferLevel::ePrimary,
                                            MAX_FRAMES_IN_FLIGHT);

    commandBuffers_ = device_.allocateCommandBuffers(allocInfo);
    imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores_.resize(swapChainImages_.size());
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    try
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            imageAvailableSemaphores_[i] = device_.createSemaphore(semaphoreInfo);

            inFlightFences_[i] = device_.createFence(fenceInfo);
        }

        for (int i = 0; i < swapChainImages_.size(); ++i)
        {
            renderFinishedSemaphores_[i] = device_.createSemaphore(semaphoreInfo);
        }
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create sync objects for a frame: ") +
                                 err.what());
    }

    spdlog::info("Sync objects created successfully");
}

void VulkanEngine::drawFrame()
{
    vk::Fence& inFlightFence = inFlightFences_[currentFrame_];
    vk::Semaphore& imageAvailableSemaphore = imageAvailableSemaphores_[currentFrame_];
    vk::CommandBuffer& commandBuffer = commandBuffers_[currentFrame_];

    (void)device_.waitForFences(1, &inFlightFence, vk::True, UINT64_MAX);
    (void)device_.resetFences(1, &inFlightFence);

    auto acquireResult =
        device_.acquireNextImageKHR(swapChain_, UINT64_MAX, imageAvailableSemaphore, nullptr);

    uint32_t imageIndex = acquireResult.value;

    vk::Semaphore& renderFinishedSemaphore = renderFinishedSemaphores_[imageIndex];

    commandBuffer.reset();

    vk::CommandBufferBeginInfo beginInfo{};

    commandBuffer.begin(beginInfo);

    // clang-format off
    vk::ImageMemoryBarrier imageBarrierToAttachment(
        {},
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        swapChainImages_[imageIndex],
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    // clang-format on
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, 0, nullptr,
                                  0, nullptr, 1, &imageBarrierToAttachment);

    vk::ClearValue clearColor(std::array<float, 4>{0.1F, 0.1F, 0.1F, 1.0F});
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.imageView = swapChainImageViews_[imageIndex];
    colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue = clearColor;

    vk::RenderingInfo renderInfo({}, {{0, 0}, swapChainExtent_}, 1, 0, 1, &colorAttachment);

    commandBuffer.beginRendering(renderInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    // clang-format off
    vk::ImageMemoryBarrier imageBarrierToPresent(
        vk::AccessFlagBits::eColorAttachmentWrite,
        {},
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        swapChainImages_[imageIndex],
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    // clang-format on

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                  vk::PipelineStageFlagBits::eBottomOfPipe, {}, 0, nullptr, 0,
                                  nullptr, 1, &imageBarrierToPresent);

    commandBuffer.end();

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo(1, &imageAvailableSemaphore, &waitStage, 1, &commandBuffer, 1,
                              &renderFinishedSemaphore);

    (void)graphicsQueue_.submit(1, &submitInfo, inFlightFence);

    vk::PresentInfoKHR presentInfo(1, &renderFinishedSemaphore, 1, &swapChain_, &imageIndex);

    (void)presentQueue_.presentKHR(presentInfo);

    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanEngine::createAllocator()
{
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = physicalDevice_;
    allocatorInfo.device = device_;
    allocatorInfo.instance = instance_;

    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&allocatorInfo, &allocator_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create VMA Allocator");
    }

    spdlog::info("VMA Allocator created successfully");
}