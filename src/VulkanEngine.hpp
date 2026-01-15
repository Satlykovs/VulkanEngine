#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <optional>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Window.hpp"

struct Vertex
{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 uv;

    bool operator==(const Vertex& other) const
    {
        return position == other.position && color == other.color && uv == other.uv;
    }

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[2].offset = offsetof(Vertex, uv);

        return attributeDescriptions;
    }
};

template <>
struct std::hash<Vertex>
{
    size_t operator()(Vertex const& vertex) const noexcept
    {
        return ((std::hash<glm::vec3>()(vertex.position) ^
                 (std::hash<glm::vec3>()(vertex.color) << 1)) >>
                1) ^
               (std::hash<glm::vec2>()(vertex.uv) << 1);
    }
};
struct AllocatedBuffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;

    glm::mat4 transform;
};

struct AllocatedImage
{
    VkImage image;
    VmaAllocation allocation;
    VkImageView imageView;
};

struct MeshPushConstants
{
    glm::mat4 renderMatrix;
};

struct SceneData
{
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
};

class VulkanEngine
{
public:
    void init(Window& window);

    void cleanup();

    void drawFrame(const SceneData& sceneData);

    void waitIdle() const
    {
        device_.waitIdle();
    }

private:
    void initVulkan();

    void createSurface();

    void pickPhysicalDevice();

    void createLogicalDevice();

    void createAllocator();

    void createDepthResources();

    void loadMeshes();
    void loadImages();

    void createTextureSampler();

    void initDescriptors();

    void createGraphicsPipeline();

    void initSyncObjects();

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
    vk::ShaderModule createShaderModule(const std::vector<char>& code) const;

    Window* window_ = nullptr;
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

    void immediateSubmit(std::function<void(vk::CommandBuffer cmd)>&& function);

    VmaAllocator allocator_;

    std::vector<Mesh> meshes_;

    AllocatedImage textureImage_;

    vk::Sampler textureSampler_;

    vk::DescriptorSetLayout descriptorSetLayout_;
    vk::DescriptorPool descriptorPool_;
    vk::DescriptorSet descriptorSet_;

    AllocatedImage depthImage_;
    vk::Format depthFormat_;

    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
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