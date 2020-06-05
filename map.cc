#include "map.h"

bool firstMouse = true;
float yaw = -90.0f; // yaw is initialized to -90.0 degrees since a yaw of 0.0
                    // results in a direction vector pointing to the right so we
                    // initially rotate a bit to the left.
float pitch = 0.0f;
float lastX = 0;
float lastY = 0;
float fov = 90.0f;

float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window);
glm::mat4 fromCV2GLM(const cv::Mat& cmat);
std::vector<float> Frustum();

// clang-format off
std::vector<float> frustumModel{
	-0.05f, 0.05f, -0.05f,
	0.05f, 0.05f, -0.05f,
	0.0f, 0.0f, 0.0f,

	0.0f, 0.0f, -0.0f,
	0.05f, 0.0f, -0.05f,
	0.05f, 0.05f, -0.05f,

	0.0f, 0.0f, -0.0f,
	-0.05f, 0.0f , -0.05f,
	-0.05f, 0.05f, -0.05f,

	0.0f, 0.0f, 0.0f,
	-0.05f, 0.0f, -0.05f,
	0.05f, 0.0f, -0.05f,

	-0.05f, 0.05f, -0.05f,
	0.05f, 0.0f, -0.05f,
	0.05f, 0.0f, -0.05f,

	-0.05f, 0.05f, -0.05f,
	0.05f, 0.05f, -0.05f,
	0.05f, 0.0f, -0.05f
};
// clang-format on

Map::Map(float _H, float _W) {
    // Initialise GLFW
    glewExperimental = true; // Needed for core profile
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }

    H = _H;
    W = _W;

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // for macOS
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(W, H, "01", NULL, NULL);
    if (window == NULL) {
        fprintf(
            stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are "
            "not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        glfwTerminate();
        return;
    }
    // Initialize GLEW
    glfwMakeContextCurrent(window);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return;
    }
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetCursorPosCallback(window, mouse_callback);

    ourShader = new Shader("camera.vs", "camera.fs");

    lastX = H / 2.0f;
    lastY = W / 2.0f;
};

void Map::prepare(std::vector<float>& p3D, std::vector<glm::mat4>& pose3d) {
    for (auto f : getFrames()) {
        cv::Mat pose = f->getPose();
        pose.at<float>(1, 3) = -pose.at<float>(1, 3);
        pose.at<float>(2, 3) = -pose.at<float>(2, 3);
        pose3d.push_back(fromCV2GLM(pose));
    }
    for (auto pt : getPoints()) {
        std::vector<float> xyz = pt->getCoords();
        std::vector<float> color = pt->getColor();
        p3D.push_back(xyz[0]);
        p3D.push_back(-xyz[1]);
        p3D.push_back(-xyz[2]);
        p3D.insert(p3D.end(), color.begin(), color.end());
    }
}

void Map::addPoint(Point* pt) {
    std::unique_lock<std::mutex> lock(mutexPoints);
    points.push_back(pt);
}

std::vector<Point*> Map::getPoints() {
    std::unique_lock<std::mutex> lock(mutexPoints);
    return points;
}

int Map::getPointsSize() {
    std::unique_lock<std::mutex> lock(mutexPoints);
    return points.size();
}

void Map::addFrame(KeyFrame* frame) {
    std::unique_lock<std::mutex> lock(mutexFrames);
    frames.push_back(frame);
}

std::vector<KeyFrame*> Map::getFrames() {
    std::unique_lock<std::mutex> lock(mutexFrames);
    return frames;
}

void Map::setCVFrame(cv::Mat frame) {
    std::unique_lock<std::mutex> lock(mutexCVFrame);
    cvFrame = frame;
}

cv::Mat Map::getCVFrame() {
    std::unique_lock<std::mutex> lock(mutexCVFrame);
    return cvFrame;
}

void Map::run() {
    while (1) {
        cv::Mat cvFrame = getCVFrame();

        if (!cvFrame.empty())
            cv::imshow("Keypoints", cvFrame);

        std::vector<float> p3D;
        std::vector<glm::mat4> pose3d;
        prepare(p3D, pose3d);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        unsigned int pointsBuffer, pointsArray;
        glGenVertexArrays(1, &pointsArray);
        glGenBuffers(1, &pointsBuffer);
        glBindVertexArray(pointsArray);

        glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
        glBufferData(GL_ARRAY_BUFFER, p3D.size() * sizeof(float), &p3D.front(),
                     GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                              (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                              (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0);

        unsigned int instanceVBO;
        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * pose3d.size(),
                     &pose3d.front(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        unsigned int frustumBuffer, frustumArray;
        glGenVertexArrays(1, &frustumArray);
        glGenBuffers(1, &frustumBuffer);
        glBindVertexArray(frustumArray);

        glBindBuffer(GL_ARRAY_BUFFER, frustumBuffer);
        glBufferData(GL_ARRAY_BUFFER, frustumModel.size() * sizeof(float),
                     &frustumModel.front(), GL_STATIC_DRAW);

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(2);

        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                              (void*)0);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                              (void*)(sizeof(glm::vec4)));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                              (void*)(2 * sizeof(glm::vec4)));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                              (void*)(3 * sizeof(glm::vec4)));

        glVertexAttribDivisor(3, 1);
        glVertexAttribDivisor(4, 1);
        glVertexAttribDivisor(5, 1);
        glVertexAttribDivisor(6, 1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // activate shader
        ourShader->use();

        // pass projection matrix to shader
        glm::mat4 projection =
            glm::perspective(glm::radians(fov), W / H, 0.1f, 10.0f);
        ourShader->setMat4("projection", projection);

        // camera/view/model transformation
        glm::mat4 view =
            glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        ourShader->setMat4("view", view);

        glm::mat4 model = glm::mat4(1.0f);
        ourShader->setMat4("model", model);

        // Bind VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
        glBindVertexArray(pointsArray);

        // Draw points
        glDrawArrays(GL_POINTS, 0, p3D.size() / 6);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Initialize and use shader
        Shader* shader1 = new Shader("frustum.vs", "frustum.fs");
        shader1->use();
        shader1->setMat4("projection", projection);
        shader1->setMat4("view", view);

        // Bind VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, frustumBuffer);
        glBindVertexArray(frustumArray);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // Draw poses
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6 * 3, pose3d.size());

        // Unbind VBO and VAO
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        glDeleteVertexArrays(1, &frustumArray);
        glDeleteBuffers(1, &frustumBuffer);
    }
}

// Handler for changing view angle
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        firstMouse = true;
        return;
    }

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    // reversed since y-coordinates go from bottom to top
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

// Handler for camera movement
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 2.5 * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -=
            glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos +=
            glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

glm::mat4 fromCV2GLM(const cv::Mat& cvmat) {
    glm::mat4 temp;
    temp[0][0] = cvmat.at<float>(0, 0);
    temp[0][1] = cvmat.at<float>(1, 0);
    temp[0][2] = cvmat.at<float>(2, 0);
    temp[0][3] = cvmat.at<float>(3, 0);
    temp[1][0] = cvmat.at<float>(0, 1);
    temp[1][1] = cvmat.at<float>(1, 1);
    temp[1][2] = cvmat.at<float>(2, 1);
    temp[1][3] = cvmat.at<float>(3, 1);
    temp[2][0] = cvmat.at<float>(0, 2);
    temp[2][1] = cvmat.at<float>(1, 2);
    temp[2][2] = cvmat.at<float>(2, 2);
    temp[2][3] = cvmat.at<float>(3, 2);
    temp[3][0] = cvmat.at<float>(0, 3);
    temp[3][1] = cvmat.at<float>(1, 3);
    temp[3][2] = cvmat.at<float>(2, 3);
    temp[3][3] = cvmat.at<float>(3, 3);

    return temp;
}
