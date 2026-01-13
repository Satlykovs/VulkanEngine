#include <iostream>

#include "VulkanEngine.hpp"
#include "spdlog/spdlog.h"

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
        spdlog::critical(e.what());
        return 1;
    }

    return 0;
}
