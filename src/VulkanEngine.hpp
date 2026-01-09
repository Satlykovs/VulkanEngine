#pragma once

#include <chrono>
#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

struct Vertex
{
    glm::vec3 position;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct MeshPushConstants
{
    glm::mat4 renderMatrix;
};

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

    void createAllocator();
    void initMesh();

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

    VmaAllocator allocator_;

    AllocatedBuffer vertexBuffer_;

    static const int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<vk::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::Fence> inFlightFences_;

    uint32_t currentFrame_ = 0;

    std::chrono::steady_clock::time_point lastFrameTime_;

    const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};