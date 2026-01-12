#pragma once

#include <glm/glm.hpp>

class Camera
{
public:
    glm::vec3 position = {0.0F, 0.0F, 40.5F};

    glm::vec3 target = {0.0F, 0.0F, 0.0F};

    glm::vec3 up = {0.0F, 1.0F, 0.0F};

    float fov = 70.0F;
    float nearPlane = 0.1F;
    float farPlane = 200.0F;

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;
};