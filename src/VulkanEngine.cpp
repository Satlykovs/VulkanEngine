#include "VulkanEngine.hpp"

#include <spdlog/spdlog.h>
#define GLFW_INCLUDE_VULKAN
#include <algorithm>
#include <fstream>
#include <glfw/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <set>

#include "stb_image.h"
#include "tiny_obj_loader.h"

void VulkanEngine::init(Window& window)
{
    spdlog::info("Initializing Engine...");
    window_ = &window;
    initVulkan();
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

        device_.destroyDescriptorPool(descriptorPool_);
        device_.destroyDescriptorSetLayout(descriptorSetLayout_);

        device_.destroyImageView(depthImage_.imageView);
        vmaDestroyImage(allocator_, depthImage_.image, depthImage_.allocation);

        device_.destroyImageView(textureImage_.imageView);
        vmaDestroyImage(allocator_, textureImage_.image, textureImage_.allocation);

        device_.destroySampler(textureSampler_);

        for (auto imageView : swapChainImageViews_)
        {
            device_.destroyImageView(imageView);
        }

        device_.destroySwapchainKHR(swapChain_);

        for (auto& mesh : meshes_)
        {
            vmaDestroyBuffer(allocator_, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);

            vmaDestroyBuffer(allocator_, mesh.indexBuffer.buffer, mesh.indexBuffer.allocation);
        }

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

        createDepthResources();

        initSyncObjects();
        loadImages();
        createTextureSampler();

        initDescriptors();

        createGraphicsPipeline();

        loadMeshes();
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
    dynamicRenderingFeatures.dynamicRendering = vk::True;

    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeeatures{};
    bufferDeviceAddressFeeatures.bufferDeviceAddress = vk::True;

    dynamicRenderingFeatures.pNext = bufferDeviceAddressFeeatures;

    vk::PhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.pNext = &dynamicRenderingFeatures;

    deviceFeatures2.features.samplerAnisotropy = vk::True;

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

    if (glfwCreateWindowSurface(instance_, window_->getNativeWindow(), nullptr, &surface) !=
        VK_SUCCESS)
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
    window_->getFrameBufferSize(width, height);

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
    createInfo.clipped = vk::True;
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

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescription = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 1, &bindingDescription,
                                                           (uint32_t)attributeDescription.size(),
                                                           attributeDescription.data());

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList,
                                                           VK_FALSE);

    vk::Viewport viewport(0.0F, 0.0F, static_cast<float>(swapChainExtent_.width),
                          static_cast<float>(swapChainExtent_.height), 0.0F, 1.0F);

    vk::Rect2D scissor({0, 0}, swapChainExtent_);

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
        vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0F, 0.0F, 0.0F, 1.0F);

    vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = vk::True;
    depthStencil.depthWriteEnable = vk::True;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = vk::False;
    depthStencil.stencilTestEnable = vk::False;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1,
                                                        &colorBlendAttachment);

    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(MeshPushConstants);
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descriptorSetLayout_, 1,
                                                    &pushConstantRange);

    try
    {
        pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create Pipeline Layout: ") + err.what());
    }

    vk::PipelineRenderingCreateInfo renderingInfo({}, 1, &swapChainImageFormat_);
    renderingInfo.depthAttachmentFormat = depthFormat_;
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
        &depthStencil,
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

void VulkanEngine::drawFrame(const SceneData& sceneData)
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

    // clang-format off
    vk::ImageMemoryBarrier depthBarrier(
        {},
        vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        vk::QueueFamilyIgnored,
        vk::QueueFamilyIgnored,
        depthImage_.image,
        {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});

    // clang-format on
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eEarlyFragmentTests, {}, 0, nullptr, 0,
                                  nullptr, 1, &depthBarrier);

    vk::ClearValue clearColor(std::array<float, 4>{0.1F, 0.1F, 0.1F, 1.0F});
    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.imageView = swapChainImageViews_[imageIndex];
    colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue = clearColor;

    vk::RenderingAttachmentInfo depthAttachment{};
    depthAttachment.imageView = depthImage_.imageView;
    depthAttachment.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    depthAttachment.clearValue.depthStencil = vk::ClearDepthStencilValue{1.0F, 0};

    vk::RenderingInfo renderInfo({}, {{0, 0}, swapChainExtent_}, 1, 0, 1, &colorAttachment,
                                 &depthAttachment, nullptr);

    commandBuffer.beginRendering(renderInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, 1,
                                     &descriptorSet_, 0, nullptr);

    for (const auto& mesh : meshes_)
    {
        vk::Buffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
        VkDeviceSize offsets[] = {0};

        commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);

        commandBuffer.bindIndexBuffer(mesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);

        MeshPushConstants constants;
        constants.renderMatrix = sceneData.projectionMatrix * sceneData.viewMatrix * mesh.transform;

        commandBuffer.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eVertex, 0,
                                    sizeof(MeshPushConstants), &constants);

        commandBuffer.drawIndexed(static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
    }

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

void VulkanEngine::loadMeshes()
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string modelPath = "../assets/models/viking_room.obj";

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str()))
    {
        throw std::runtime_error(warn + err);
    }

    for (const auto& shape : shapes)
    {
        Mesh newMesh;
        newMesh.transform = glm::mat4(1.0F);

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& index : shape.mesh.indices)
        {
            Vertex vertex{};
            vertex.position = {attrib.vertices[3 * index.vertex_index + 0],
                               attrib.vertices[3 * index.vertex_index + 1],
                               attrib.vertices[3 * index.vertex_index + 2]};

            vertex.color = vertex.position;  // TEMPORARY

            vertex.uv = {attrib.texcoords[2 * index.texcoord_index + 0],
                         attrib.texcoords[2 * index.texcoord_index +
                                          1]};  // It works without 1.0f - ... (strange model =))

            if (!uniqueVertices.contains(vertex))
            {
                uniqueVertices[vertex] = static_cast<uint32_t>(newMesh.vertices.size());

                newMesh.vertices.push_back(vertex);
            }
            newMesh.indices.push_back(uniqueVertices[vertex]);
        }

        size_t bufferSize = newMesh.vertices.size() * sizeof(Vertex);

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;

        VmaAllocationCreateInfo vmaAllocInfo{};
        vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        if (vmaCreateBuffer(allocator_, (VkBufferCreateInfo*)&bufferInfo, &vmaAllocInfo,
                            &newMesh.vertexBuffer.buffer, &newMesh.vertexBuffer.allocation,
                            &newMesh.vertexBuffer.info) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate vertex buffer for mesh");
        }

        void* data;
        vmaMapMemory(allocator_, newMesh.vertexBuffer.allocation, &data);
        memcpy(data, newMesh.vertices.data(), bufferSize);
        vmaUnmapMemory(allocator_, newMesh.vertexBuffer.allocation);

        size_t indexBufferSize = newMesh.indices.size() * sizeof(uint32_t);

        vk::BufferCreateInfo indexBufferInfo{};
        indexBufferInfo.size = indexBufferSize;
        indexBufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;

        if (vmaCreateBuffer(allocator_, (VkBufferCreateInfo*)&indexBufferInfo, &vmaAllocInfo,
                            &newMesh.indexBuffer.buffer, &newMesh.indexBuffer.allocation,
                            &newMesh.indexBuffer.info) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate index buffer");
        }

        vmaMapMemory(allocator_, newMesh.indexBuffer.allocation, &data);
        memcpy(data, newMesh.indices.data(), indexBufferSize);
        vmaUnmapMemory(allocator_, newMesh.indexBuffer.allocation);

        meshes_.push_back(newMesh);
        spdlog::info("Loaded shape [{}]: {} vertices, {} indices", shape.name,
                     newMesh.vertices.size(), newMesh.indices.size());
    }

    spdlog::info("Model loading complete. Total objects: {}", meshes_.size());
}

void VulkanEngine::createDepthResources()
{
    depthFormat_ = vk::Format::eD32Sfloat;

    vk::Extent3D depthImageExtent = {swapChainExtent_.width, swapChainExtent_.height, 1};

    vk::ImageCreateInfo dImgInfo{};

    dImgInfo.imageType = vk::ImageType::e2D;
    dImgInfo.format = depthFormat_;
    dImgInfo.extent = depthImageExtent;
    dImgInfo.mipLevels = 1;
    dImgInfo.arrayLayers = 1;
    dImgInfo.samples = vk::SampleCountFlagBits::e1;
    dImgInfo.tiling = vk::ImageTiling::eOptimal;
    dImgInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

    VmaAllocationCreateInfo dImgAllocInfo{};
    dImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dImgAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vmaCreateImage(allocator_, (VkImageCreateInfo*)&dImgInfo, &dImgAllocInfo,
                       &depthImage_.image, &depthImage_.allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate depth image");
    }

    vk::ImageViewCreateInfo dViewInfo{};
    dViewInfo.viewType = vk::ImageViewType::e2D;
    dViewInfo.image = depthImage_.image;
    dViewInfo.format = depthFormat_;
    dViewInfo.subresourceRange.baseMipLevel = 0;
    dViewInfo.subresourceRange.levelCount = 1;
    dViewInfo.subresourceRange.baseArrayLayer = 0;
    dViewInfo.subresourceRange.layerCount = 1;
    dViewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;

    try
    {
        depthImage_.imageView = device_.createImageView(dViewInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create depth image view: ") + err.what());
    }

    spdlog::info("Depth resources created successfully");
}

void VulkanEngine::loadImages()
{
    int texWidth, texHeight, texChannels;

    stbi_uc* pixels = stbi_load("../assets/textures/viking_room.png", &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);

    if (!pixels)
    {
        throw std::runtime_error("Failed to load texture image");
    }

    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(texWidth) * texHeight * 4;

    AllocatedBuffer stagingBuffer;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = imageSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

    VmaAllocationCreateInfo vmaAllocInfo{};

    vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    if (vmaCreateBuffer(allocator_, (VkBufferCreateInfo*)&bufferInfo, &vmaAllocInfo,
                        &stagingBuffer.buffer, &stagingBuffer.allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create staging buffer for texture");
    }

    void* data;
    vmaMapMemory(allocator_, stagingBuffer.allocation, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vmaUnmapMemory(allocator_, stagingBuffer.allocation);

    stbi_image_free(pixels);

    vk::Extent3D imageExtent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight),
                                1};

    vk::ImageCreateInfo dImgInfo{};

    dImgInfo.imageType = vk::ImageType::e2D;
    dImgInfo.format = vk::Format::eR8G8B8A8Srgb;
    dImgInfo.extent = imageExtent;
    dImgInfo.mipLevels = 1;
    dImgInfo.arrayLayers = 1;
    dImgInfo.samples = vk::SampleCountFlagBits::e1;
    dImgInfo.tiling = vk::ImageTiling::eOptimal;
    dImgInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

    VmaAllocationCreateInfo dImgAllocInfo{};
    dImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(allocator_, (VkImageCreateInfo*)&dImgInfo, &dImgAllocInfo,
                       &textureImage_.image, &textureImage_.allocation, nullptr) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate texture image");
    }

    spdlog::info("Texture loaded to staging buffer and memory allocated");

    immediateSubmit(
        [&](vk::CommandBuffer cmd)
        {
            // clang-format off

            vk::ImageMemoryBarrier barrierToTransfer(
           {},
           vk::AccessFlagBits::eTransferWrite,
           vk::ImageLayout::eUndefined,
           vk::ImageLayout::eTransferDstOptimal,
           vk::QueueFamilyIgnored,
           vk::QueueFamilyIgnored,
           textureImage_.image,
           {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});


            cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {},
            0, nullptr,
            0, nullptr,
            1, &barrierToTransfer
            );

            // clang-format on

            vk::BufferImageCopy copyRegion{};
            copyRegion.bufferOffset = 0;
            copyRegion.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            copyRegion.imageExtent = imageExtent;

            cmd.copyBufferToImage(stagingBuffer.buffer, textureImage_.image,
                                  vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

            // clang-format off

            vk::ImageMemoryBarrier barrierToShader(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::QueueFamilyIgnored,
                vk::QueueFamilyIgnored,
                textureImage_.image,
                {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader,
                {},
                0, nullptr,
                0, nullptr,
                1, &barrierToShader
                );
            // clang-format on
        });

    vmaDestroyBuffer(allocator_, stagingBuffer.buffer, stagingBuffer.allocation);

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = textureImage_.image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = vk::Format::eR8G8B8A8Srgb;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    try
    {
        textureImage_.imageView = device_.createImageView(viewInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create texture image view: ") + err.what());
    }

    spdlog::info("Texture loaded successfully");
}

void VulkanEngine::immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function)
{
    vk::CommandBufferAllocateInfo allocInfo(commandPool_, vk::CommandBufferLevel::ePrimary, 1);

    vk::CommandBuffer cmd = device_.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);

    function(cmd);
    cmd.end();

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &cmd);

    graphicsQueue_.submit(submitInfo);
    graphicsQueue_.waitIdle();

    device_.freeCommandBuffers(commandPool_, 1, &cmd);
}

void VulkanEngine::createTextureSampler()
{
    vk::PhysicalDeviceProperties properties = physicalDevice_.getProperties();

    vk::SamplerCreateInfo samplerInfo{};

    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;

    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;

    samplerInfo.anisotropyEnable = vk::True;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;

    samplerInfo.unnormalizedCoordinates = vk::False;

    samplerInfo.compareEnable = vk::False;

    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

    try
    {
        textureSampler_ = device_.createSampler(samplerInfo);
    }
    catch (const vk::SystemError& err)
    {
        throw std::runtime_error(std::string("Failed to create texture sampler:") + err.what());
    }
}

void VulkanEngine::initDescriptors()
{
    vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, 1, &samplerLayoutBinding);
    descriptorSetLayout_ = device_.createDescriptorSetLayout(layoutInfo);

    vk::DescriptorPoolSize poolSize(vk::DescriptorType::eCombinedImageSampler, 1);

    vk::DescriptorPoolCreateInfo poolInfo({}, 1, 1, &poolSize);
    descriptorPool_ = device_.createDescriptorPool(poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool_, 1, &descriptorSetLayout_);
    descriptorSet_ = device_.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorImageInfo imageInfo{};

    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = textureImage_.imageView;
    imageInfo.sampler = textureSampler_;

    vk::WriteDescriptorSet descriptorWrite{};

    descriptorWrite.dstSet = descriptorSet_;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;

    device_.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
}
