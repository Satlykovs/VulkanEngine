#include "Camera.hpp"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera()
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const
{
    glm::mat4 projection = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
    projection[1][1] *= -1;
    return projection;
}

void Camera::processKeyboard(CameraMovement direction, float deltaTime)
{
    float velocity = movementSpeed * deltaTime;

    if (direction == CameraMovement::FORWARD) position += front * velocity;
    if (direction == CameraMovement::BACKWARD) position -= front * velocity;
    if (direction == CameraMovement::LEFT) position -= right * velocity;
    if (direction == CameraMovement::RIGHT) position += right * velocity;
    if (direction == CameraMovement::UP) position += worldUp * velocity;
    if (direction == CameraMovement::DOWN) position -= worldUp * velocity;
}

void Camera::processMouseMovement(float xOffset, float yOffset, bool constrainPitch)
{
    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (constrainPitch)
    {
        pitch = std::clamp(pitch, -89.0F, 89.0F);
    }

    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    glm::vec3 newFront;
    newFront.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    newFront.y = sinf(glm::radians(pitch));
    newFront.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    front = glm::normalize(newFront);

    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}
