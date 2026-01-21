#include "Window.hpp"

#include <stdexcept>

Window::Window(int width, int height, std::string title)
    : width_(width), height_(height), title_(std::move(title))
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);

    if (!window_)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }

    glfwSetWindowUserPointer(window_, this);

    glfwSetFramebufferSizeCallback(window_, frameBufferResizedCallback);
}

Window::~Window()
{
    if (window_)
    {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

void Window::frameBufferResizedCallback(GLFWwindow* window, int width, int height)
{
    auto appWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    if (appWindow)
    {
        appWindow->frameBufferResized_ = true;
    }
}
