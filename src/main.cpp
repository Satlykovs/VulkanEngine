#include "Camera.hpp"
#include "VulkanEngine.hpp"
#include "Window.hpp"
#include "spdlog/spdlog.h"

float lastX = 400;
float lastY = 300;
bool firstMouse = true;

struct KeyAction
{
    int key;
    CameraMovement movement;
};

static const std::array<KeyAction, 6> keyMappings = {
    {{.key = GLFW_KEY_W, .movement = CameraMovement::FORWARD},
     {.key = 83, .movement = CameraMovement::BACKWARD},
     {.key = GLFW_KEY_A, .movement = CameraMovement::LEFT},
     {.key = GLFW_KEY_D, .movement = CameraMovement::RIGHT},
     {.key = GLFW_KEY_Q, .movement = CameraMovement::DOWN},
     {.key = GLFW_KEY_E, .movement = CameraMovement::UP}}};

void processInput(Window& window, Camera& camera, float deltaTime)
{
    GLFWwindow* nativeWin = window.getNativeWindow();

    if (glfwGetMouseButton(nativeWin, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        window.setMouseCapture(true);

        double xPos;
        double yPos;
        glfwGetCursorPos(nativeWin, &xPos, &yPos);

        if (firstMouse)
        {
            lastX = static_cast<float>(xPos);
            lastY = static_cast<float>(yPos);
            firstMouse = false;
        }

        float xOffset = static_cast<float>(xPos) - lastX;
        float yOffset = static_cast<float>(yPos) - lastY;

        lastX = static_cast<float>(xPos);
        lastY = static_cast<float>(yPos);

        camera.processMouseMovement(xOffset, yOffset);

        if (glfwGetKey(nativeWin, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.movementSpeed = 10.0F;
        else
            camera.movementSpeed = 2.5F;

        for (const auto& mapping : keyMappings)
        {
            if (glfwGetKey(nativeWin, mapping.key) == GLFW_PRESS)
            {
                camera.processKeyboard(mapping.movement, deltaTime);
            }
        }
    }
    else
    {
        window.setMouseCapture(false);
        firstMouse = true;
    }
}

int main()
{
    try
    {
        Window window(800, 600, "Vulkan Engine");

        VulkanEngine engine;
        engine.init(window);

        Camera camera;

        auto lastTime = std::chrono::steady_clock::now();

        while (!window.shouldClose())
        {
            Window::pollEvents();

            auto currentTime = std::chrono::steady_clock::now();

            int width;
            int height;

            window.getFrameBufferSize(width, height);
            if (width == 0 || height == 0) continue;

            float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

            float deltaTime =
                static_cast<std::chrono::duration<float>>(currentTime - lastTime).count();

            lastTime = currentTime;

            processInput(window, camera, deltaTime);

            SceneData sceneData{};
            sceneData.viewMatrix = camera.getViewMatrix();
            sceneData.projectionMatrix = camera.getProjectionMatrix(aspectRatio);

            engine.drawFrame(sceneData);
        }

        engine.waitIdle();
        engine.cleanup();
    }
    catch (const std::exception& e)
    {
        spdlog::critical(e.what());
        return 1;
    }

    return 0;
}
