#ifndef MAP_H
#define MAP_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>


#include <opencv2/core/core.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "shader.h"


class Map {

public:
    Map();
    void run(std::vector<float>& p3D);
private:
    GLFWwindow* window;
    Shader* ourShader; 
};

#endif // INIT_H
