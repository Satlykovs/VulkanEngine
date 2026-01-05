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

    void createGraphicsPipeline();

    void initSyncObjects();

    void drawFrame();

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

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;

        std::vector<vk::SurfaceFormatKHR> formats;

        std::vector<vk::PresentModeKHR> presentModes;
    };

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) const;

    [[nodiscard]] SwapChainSupportDetails querySwapChainSupport(
        const vk::PhysicalDevice& device) const;

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& availableFormats);

    static vk::PresentModeKHR chooseSwapPresentMode(
        const std::vector<vk::PresentModeKHR>& availablePresentModes);

    [[nodiscard]] vk::Extent2D chooseSwapExtent(
        const vk::SurfaceCapabilitiesKHR& capabilities) const;

    void createSwapChain();
    void createImageViews();

    static std::vector<char> readFile(const std::string& filename);
    vk::ShaderModule createShaderModule(const std::vector<char>& code);

    const int WIDTH = 800;
    const int HEIGHT = 600;

    GLFWwindow* window_ = nullptr;

    vk::Instance instance_;
    vk::SurfaceKHR surface_;

    vk::PhysicalDevice physicalDevice_ = nullptr;

    vk::Device device_;

    vk::Queue graphicsQueue_;
    vk::Queue presentQueue_;

    vk::SwapchainKHR swapChain_;
    std::vector<vk::Image> swapChainImages_;
    vk::Format swapChainImageFormat_;
    vk::Extent2D swapChainExtent_;
    std::vector<vk::ImageView> swapChainImageViews_;

    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline graphicsPipeline_;

    vk::CommandPool commandPool_;

    std::vector<vk::CommandBuffer> commandBuffers_;

    static const int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;

    uint32_t currentFrame_ = 0;

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};