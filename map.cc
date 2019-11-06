#include "map.h"

float H = 1280.0f * 0.75f;
float W = 720.0f * 0.75f;

bool firstMouse = true;
float yaw   = -90.0f;	// yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch =  0.0f;
float lastX =  H / 2.0f;
float lastY =  W / 2.0f;
float fov   =  90.0f;

float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);
std::vector<float> Frustum();

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

Map::Map() {
    // Initialise GLFW
    glewExperimental = true; // Needed for core profile
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return;
    }

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 
	window = glfwCreateWindow(H, W, "01", NULL, NULL);
	if( window == NULL ){ fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
	    glfwTerminate();
	    return;
	}
    glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental=true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
	    fprintf(stderr, "Failed to initialize GLEW\n");
	    return;
	}
    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetCursorPosCallback(window, mouse_callback);

    ourShader = new Shader("camera.vs", "camera.fs");
};

void Map::run(std::vector<float>& p3D, std::vector<glm::vec3>& pose3d) {

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    unsigned int pointsBuffer, pointsArray;
    glGenVertexArrays(1, &pointsArray);
    glGenBuffers(1, &pointsBuffer);
    glBindVertexArray(pointsArray);

    glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
    glBufferData(GL_ARRAY_BUFFER, p3D.size() * sizeof(float), &p3D.front(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    glBindVertexArray(0); 

	unsigned int instanceVBO;
	glGenBuffers(1, &instanceVBO);
	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * pose3d.size(), &pose3d.front(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    unsigned int frustumBuffer, frustumArray;
    glGenVertexArrays(1, &frustumArray);
    glGenBuffers(1, &frustumBuffer);
    glBindVertexArray(frustumArray);

    glBindBuffer(GL_ARRAY_BUFFER, frustumBuffer);
    glBufferData(GL_ARRAY_BUFFER, frustumModel.size() * sizeof(float), &frustumModel.front(), GL_STATIC_DRAW);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
	
	glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO); 
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glVertexAttribDivisor(2,1);
    glBindVertexArray(0); 
	
	
    //do {
		
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
        processInput(window);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        // activate shader
        ourShader->use();

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(fov), H / W, 0.1f, 10.0f);
        ourShader->setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        ourShader->setMat4("view", view);
        
        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        ourShader->setMat4("model", model);
        
        glBindBuffer(GL_ARRAY_BUFFER, pointsBuffer);
        glBindVertexArray(pointsArray);
        glDrawArrays(GL_POINTS, 0, p3D.size());
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        
        Shader* shader1 = new Shader("frustum.vs", "frustum.fs");
        
        shader1->use();
        shader1->setMat4("projection", projection);
        shader1->setMat4("view", view);
        shader1->setMat4("model", model);

        glBindBuffer(GL_ARRAY_BUFFER, frustumBuffer);
        glBindVertexArray(frustumArray);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        
		glDrawArraysInstanced(GL_TRIANGLES, 0, 6 * 3, pose3d.size());

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        
        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
		
		glDeleteVertexArrays(1, &frustumArray);
		glDeleteBuffers(1, &frustumBuffer);

    //} //while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 );
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) 
  {
      firstMouse = true;
      return;
  }

  if (firstMouse)
  {
      lastX = xpos;
      lastY = ypos;
      firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
  lastX = xpos;
  lastY = ypos;

  float sensitivity = 0.1f; // change this value to your liking
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


void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 2.5 * deltaTime; 
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

std::vector<float> Frustum() {

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
      
    return frustumModel;
}
