#include <iostream>

#include "VulkanEngine.hpp"

int main()
{
    VulkanEngine engine;

    try
    {
        engine.init();
        engine.run();
        engine.cleanup();
    }
    catch (const std::exception& e)
    {
        std::cerr << "CRITICAL ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
