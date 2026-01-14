#pragma once

#include <glm/glm.hpp>

enum class CameraMovement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera
{
public:
    glm::vec3 position = {0.0F, 0.0F, 40.5F};

    glm::vec3 front = {0.0F, 0.0F, -1.0F};
    glm::vec3 up = {0.0F, 1.0F, 0.0F};
    glm::vec3 right = {1.0F, 0.0F, 0.0F};
    glm::vec3 worldUp = {0.0F, 1.0F, 0.0F};

    float yaw = -90.0F;
    float pitch = 0.0F;

    float movementSpeed = 2.5F;
    float mouseSensitivity = 0.1F;
    float fov = 70.0F;
    float nearPlane = 0.1F;
    float farPlane = 200.0F;

    Camera();

    [[nodiscard]] glm::mat4 getViewMatrix() const;
    [[nodiscard]] glm::mat4 getProjectionMatrix(float aspectRatio) const;

    void processKeyboard(CameraMovement direction, float deltaTime);
    void processMouseMovement(float xOffset, float yOffset, bool constrainPitch = true);

private:
    void updateCameraVectors();
};