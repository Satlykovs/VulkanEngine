#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>

class Window
{
public:
    Window(int width, int height, std::string title);

    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool shouldClose() const
    {
        return glfwWindowShouldClose(window_);
    }

    static void pollEvents()
    {
        glfwPollEvents();
    }

    static void waitEvents()
    {
        glfwWaitEvents();
    }

    [[nodiscard]] GLFWwindow* getNativeWindow() const
    {
        return window_;
    }

    void getFrameBufferSize(int& width, int& height) const
    {
        glfwGetFramebufferSize(window_, &width, &height);
    }

    void setMouseCapture(bool captured)
    {
        if (captured)
        {
            glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else
        {
            glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }

    bool wasWindowResized()
    {
        return frameBufferResized_;
    }

    void resetWindowResizedFlag()
    {
        frameBufferResized_ = false;
    }

private:
    GLFWwindow* window_;
    int width_;
    int height_;
    std::string title_;

    bool frameBufferResized_ = false;

    static void frameBufferResizedCallback(GLFWwindow* window, int width, int height);
};