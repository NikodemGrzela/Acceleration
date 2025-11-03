#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <shader_m.h>
#include <stb_image.h>
#include <camera.h>
#include <ComputeShader.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <cmath>
#include <map>  
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <print>

#define N_FUNCTIONS 3
#define N_DIMENSIONS 2
#define N_VARS 4

void imgui_init(float main_scale);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void shaderCompilation();
void shaderCompilationStatus(unsigned int shader_id);
void startup();
void render();
unsigned int loadTexture(char const * path);
void texture_setup();

void compute_bransley(float *vertices);
void compute_sierpinski(float *vertices);
void compute_threaded(float * vertices);
void thread_compute(float *vertices, unsigned int  start, unsigned int  end);
void compute_random(float *vertices);
void compute_random_but_cooler(float *vertices);
void compute_with_shader(float *vertices,ComputeShader &computeShader);
void compute_bransley_shader(float *vertices,ComputeShader &computeShader);

void renderQuad();
float generateFloat();
void fill_transform();
void fill_bransley();
void fill_sierpinski();
void imgui_matrix(unsigned int transform_number, const char *name);
void generate_points(float *vertices);



// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;



// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

bool mouseCaptured = true;
bool just_transformed = false;

unsigned int number_of_points = 2000000;
unsigned int number_of_vertices = number_of_points*4;
unsigned int iterations = 10;

float low = 0.0f;
float high = 1.0f;

int number_of_threads = 32;
thread_local std::mt19937 generator;
thread_local std::uniform_int_distribution<int> distribution(0, 2); 
std::uniform_real_distribution<> dis(-1.0, 1.0); 



bool gpu_compute = false;

float random_transform[N_FUNCTIONS][N_DIMENSIONS][N_VARS];
glm::mat4 transforms[N_FUNCTIONS];
glm::mat4 bransley_transform[4];
glm::mat4 sierpinski_transform[3];

unsigned int VBO, VAO;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    
    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor()); // Valid on GLFW 3.3+ only
    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Akceleracja", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback); 

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  
    
    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }    
    


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; 

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale; 

    ImGui_ImplGlfw_InitForOpenGL(window, true);

    ImGui_ImplOpenGL3_Init("#version 130");

    glEnable(GL_PROGRAM_POINT_SIZE);  
    
    // build and compile our shader zprogram
    // ------------------------------------
    Shader shader("shaders/shader.vs", "shaders/shader.fs");
    Shader screenQuad("shaders/quad.vs", "shaders/quad.fs");
    ComputeShader computeShader("shaders/shader.comp");   
    
   

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    fill_transform();
    
    float *vertices = new float[number_of_vertices]; 
    
    generate_points(vertices);
    fill_sierpinski();
    fill_bransley();
    

    glCreateBuffers(1, &VBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, VBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * number_of_vertices, vertices, GL_DYNAMIC_DRAW);
    
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0); 


    // draw as wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 model;

    shader.use();
    //compute_bransley(vertices);
    //std::println("{} \n", glm::to_string(transforms[0]));
    // render loop
    // -----------
    bool show_matrix1 = false;
    bool show_matrix2 = false;
    bool show_matrix3 = false;

    bool cpu = true;
    bool cpu_threaded = false;
    bool gpu = false;
    

    static bool ref_color = false;
    static ImVec4 ref_color_v(1.0f, 0.0f, 1.0f, 0.5f);
    static ImGuiColorEditFlags flags = ImGuiColorEditFlags_AlphaBar;
    flags |= ImGuiColorEditFlags_PickerHueWheel;
    flags |= ImGuiColorEditFlags_DisplayRGB;    
    flags |= ImGuiColorEditFlags_NoSidePreview;
    static ImVec4 color = ImVec4(114.0f / 255.0f, 144.0f / 255.0f, 154.0f / 255.0f, 200.0f / 255.0f);

    int fCounter = 0;
    while (!glfwWindowShouldClose(window))
    {
        // timing 
       
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        

        // Set frame time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // if(fCounter > 500) {
        //         std::println("FPS {}", 1/deltaTime);
        //         fCounter = 0;
        // } else {
        //     fCounter++;
        // }	

        // input
        // -----
        //if (!io.MouseHoveredViewport) 
        processInput(window);
        

        
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(color.x/2, color.y/2, color.z/2, 1.0f);

        
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.0f, 100.0f);
        view = camera.GetViewMatrix();
        model = glm::mat4(1.0f); 
        
        model = glm::translate(model, glm::vec3(2.0f, 0.0f, 0.0f)); 
        model = glm::scale(model, glm::vec3(1.0f)); 

        shader.setVec4("color", glm::vec4(color.x,color.y,color.z,color.w));
        shader.setMat4("projection", projection);
        shader.setMat4("view", view);
        shader.setMat4("model", model);
        //compute_bransley(vertices);
        if(cpu) compute_sierpinski(vertices); 
        if(cpu_threaded) compute_threaded(vertices);
        //compute_random_but_cooler(vertices);
        if(gpu) compute_with_shader(vertices, computeShader);
        //compute_bransley_shader(vertices,computeShader);
        shader.use();
        if(!gpu){
            glBindBuffer(GL_ARRAY_BUFFER, VBO); 
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * number_of_vertices, vertices, GL_DYNAMIC_DRAW);
        }
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, number_of_points);
        
        
        ImGui::Begin("Tools");
        ImGui::Checkbox("Transform 1", &show_matrix1);
        ImGui::SameLine();
        ImGui::Checkbox("Transform 2", &show_matrix2);
        ImGui::SameLine();
        ImGui::Checkbox("Transform 3", &show_matrix3);
        ImGui::NewLine();
        ImGui::Checkbox("CPU", &cpu);
        ImGui::SameLine();
        ImGui::Checkbox("CPU Threaded", &cpu_threaded);
        ImGui::SameLine();
        ImGui::Checkbox("GPU", &gpu);
        ImGui::ColorPicker4("MyColor##4", (float*)&color, flags, ref_color ? &ref_color_v.x : NULL);
        ImGui::NewLine();
        if(show_matrix1) imgui_matrix(0, "Transform 1");
        if(show_matrix2) imgui_matrix(1, "Transform 2");
        if(show_matrix3) imgui_matrix(2, "Transform 3");
        if(ImGui::Button("Randomize!")) fill_transform();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        
        ImGui::End();

        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);


    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && mouseCaptured) {
        mouseCaptured = !mouseCaptured;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);  
    }
    else if(glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS && !mouseCaptured) {
        mouseCaptured = !mouseCaptured;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !just_transformed) {
        just_transformed = true;
        fill_transform();
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS && just_transformed) {
        just_transformed = false;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn){

    if(!mouseCaptured) return;
    
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; 

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void imgui_init(float main_scale){
    
}

void imgui_matrix(unsigned int transform_number, const char *name){
    ImGui::Begin(name);
        if (ImGui::BeginTable("Matrix", 4))
        {
            unsigned int id = 0;
            for (int row = 0; row < 4; row++)
            {
                ImGui::TableNextRow();
                for (int column = 0; column < 4; column++)
                {
                    ImGui::TableSetColumnIndex(column);
                    ImGui::PushID(id);
                    ImGui::SliderFloat(" ", &transforms[transform_number][row][column], -1.0f, 1.0f);
                    ImGui::PopID();
                    id++;
                }
            }
            ImGui::EndTable();
        }        
        ImGui::End();
}


void compute_bransley(float *vertices){
    for(int i = 0; i < iterations; i++){
        for(unsigned int j = 0; j < number_of_vertices; j+=4){
            int choice = rand() % 4;
            float V_x = vertices[j];
            float V_y = vertices[j+1];
            if (choice == 0){    
                vertices[j] = 0.85f * V_x + 0.04f * V_y;
                vertices[j + 1] = -0.04f * V_x + 0.85f * V_y + 1.60f;
            }
            if (choice == 1){
                vertices[j] = -0.15f * V_x + 0.28f * V_y;
                vertices[j + 1] = 0.26f * V_x + 0.24f * V_y + 0.44f;
            }
            if (choice == 2){
                vertices[j] = 0.20f * V_x - 0.26f * V_y;
                vertices[j + 1] = 0.23f * V_x + 0.22f * V_y + 1.60f;
            }
            if (choice == 3){
                vertices[j] = 0.0f;
                vertices[j + 1] = 0.16f * V_y;
            }
        }    
    }
}

void compute_threaded(float *vertices){
    std::vector<std::thread> threads;
    threads.reserve(number_of_threads);

    for(int i = 0; i < iterations; i++){
        threads.clear();
        for(int j = 0; j < number_of_threads; j++){
            unsigned int start = (number_of_vertices/number_of_threads) * j;
            unsigned int end = (number_of_vertices/number_of_threads) * (j + 1);
            threads.emplace_back(std::thread(thread_compute, vertices, start, end));
        }
        for (auto& thread : threads) {
            if(thread.joinable()) thread.join();
        }    
    }
}

void thread_compute(float *vertices, unsigned int  start, unsigned int  end){
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count() + start);

    for(int j = start; j < end; j+=4){
            int choice = distribution(generator);
            float V_x = vertices[j];
            float V_y = vertices[j+1];
            if (choice == 0){    
                vertices[j] = V_x/2;
                vertices[j + 1] = V_y/2 + 0.36f;
            }
            if (choice == 1){
                vertices[j] = V_x/2 - 0.5f;
                vertices[j + 1] = V_y/2 - 0.5f;
            }
            if (choice == 2){
                vertices[j] = V_x/2 + 0.5f;
                vertices[j + 1] = V_y/2 - 0.5f;
            }
            // if (choice == 0){    
            //     vertices[j] = 0.85f * V_x + 0.04f * V_y;
            //     vertices[j + 1] = -0.04f * V_x + 0.85f * V_y + 1.60f;
            // }
            // if (choice == 1){
            //     vertices[j] = -0.15f * V_x + 0.28f * V_y;
            //     vertices[j + 1] = 0.26f * V_x + 0.24f * V_y + 0.44f;
            // }
            // if (choice == 2){
            //     vertices[j] = 0.20f * V_x - 0.26f * V_y;
            //     vertices[j + 1] = 0.23f * V_x + 0.22f * V_y + 1.60f;
            // }
            // if (choice == 3){
            //     vertices[j] = 0.0f;
            //     vertices[j + 1] = 0.16f * V_y;
            // }
        }
}

void compute_sierpinski(float *vertices){
    for(int i = 0; i < iterations; i++){
        for(unsigned int j = 0; j < number_of_vertices; j+=4){
            int choice = rand() % 3;
            float V_x = vertices[j];
            float V_y = vertices[j+1];
            if (choice == 0){    
                vertices[j] = V_x/2;
                vertices[j + 1] = V_y/2 + 0.36f;
            }
            if (choice == 1){
                vertices[j] = V_x/2 - 0.5f;
                vertices[j + 1] = V_y/2 - 0.5f;
            }
            if (choice == 2){
                vertices[j] = V_x/2 + 0.5f;
                vertices[j + 1] = V_y/2 - 0.5f;
            }
        }
        
    }
}

void compute_random(float *vertices){
    for(int i = 0; i < iterations; i++){
        for(unsigned int j = 0; j < number_of_vertices; j+=4){
            int choice = rand() % 3;
            float V_x = vertices[j];
            float V_y = vertices[j+1];
            //float V_z = vertices[j+2];
            if (choice == 0){    
                vertices[j]     = (V_x * transforms[choice][0][0] + transforms[choice][0][3]);
                vertices[j + 1] = (V_y * transforms[choice][1][1] + transforms[choice][1][3]);
                //vertices[j + 2] = (V_z * transforms[choice][2][2] + transforms[choice][2][3]);
            }
            if (choice == 1){
                vertices[j]     = (V_x * transforms[choice][0][0] + transforms[choice][0][3]);
                vertices[j + 1] = (V_y * transforms[choice][1][1] + transforms[choice][1][3]);
                //vertices[j + 2] = (V_z * transforms[choice][2][2] + transforms[choice][2][3]);
            }
            if (choice == 2){
                vertices[j]     = (V_x * transforms[choice][0][0] + transforms[choice][0][3]);
                vertices[j + 1] = (V_y * transforms[choice][1][1] + transforms[choice][1][3]);
                //vertices[j + 2] = (V_z * transforms[choice][2][2] + transforms[choice][2][3]);
            }
        }
    }
}

void compute_random_but_cooler(float *vertices){
    for(int i = 0; i < iterations; i++){
        for(unsigned int j = 0; j < number_of_vertices; j+=4){
            int choice = rand() % 3;
            float V_x = vertices[j];
            float V_y = vertices[j+1];
            //float V_z = vertices[j+2];
            if (choice == 0){    
                vertices[j] = (V_x *     random_transform[choice][0][0] + random_transform[choice][0][1]) 
                            + (V_y *     random_transform[choice][0][2] + random_transform[choice][0][3]);
                vertices[j + 1] = (V_x * random_transform[choice][1][0] + random_transform[choice][1][1]) 
                            + (V_y *     random_transform[choice][1][2] + random_transform[choice][1][3]);
            }
            if (choice == 1){
                vertices[j] = (V_x *     random_transform[choice][0][0] + random_transform[choice][0][1]) 
                            + (V_y *     random_transform[choice][0][2] + random_transform[choice][0][3]);
                vertices[j + 1] = (V_x * random_transform[choice][1][0] + random_transform[choice][1][1]) 
                            + (V_y *     random_transform[choice][1][2] + random_transform[choice][1][3]);
            }
            if (choice == 2){
                vertices[j] = (V_x *     random_transform[choice][0][0] + random_transform[choice][0][1]) 
                            + (V_y *     random_transform[choice][0][2] + random_transform[choice][0][3]);
                vertices[j + 1] = (V_x * random_transform[choice][1][0] + random_transform[choice][1][1]) 
                            + (V_y *     random_transform[choice][1][2] + random_transform[choice][1][3]);
            }
        }   
    }
}

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
	if (quadVAO == 0)
	{
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

float generateFloat(){
    return -1.0f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.0f- (-1.0f))));
}

void fill_transform(){
    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(-1.0, 1.0); 
    for( int i = 0; i < N_FUNCTIONS; i++) transforms[i] = glm::mat4(1.0f);
    for(int i = 0; i < N_FUNCTIONS; i++){
        for(int j = 0; j < N_DIMENSIONS; j++){
            for(int k = 0; k < N_VARS; k++) {
                random_transform[i][j][k] = (float)dis(generator);
            }
        }
    }
    for(int i = 0; i < N_FUNCTIONS; i++){
        transforms[i] = glm::mat4(glm::vec4((float)dis(generator),0.0f,0.0f,(float)dis(generator)),
                                  glm::vec4(0.0f,(float)dis(generator),0.0f,(float)dis(generator)),
                                  glm::vec4(0.0f,0.0f,(float)dis(generator),(float)dis(generator)),
                                  glm::vec4(0.0f,0.0f,0.0f,1.0f));
        // transforms[i] = glm::mat4(glm::vec4((float)dis(gen),0.0f,0.0f,0.0f),
        //                           glm::vec4(0.0f,(float)dis(gen),0.0f,0.0f),
        //                           glm::vec4(0.0f,0.0f,1.0f,0.0f),
        //                           glm::vec4((float)dis(gen),(float)dis(gen),0.0f,1.0f));
    }
}

void fill_bransley(){
    bransley_transform[0] = glm::mat4(glm::vec4(0.85f,0.04f,0.0f,0.0f),
                                      glm::vec4(-0.04f,0.85f,0.0f,1.60f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
    bransley_transform[1] = glm::mat4(glm::vec4(-0.15f,0.28f,0.0f,0.0f),
                                      glm::vec4(0.26f,0.24f,0.0f,0.44f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
    bransley_transform[2] = glm::mat4(glm::vec4(0.20f,-0.26f,0.0f,0.0f),
                                      glm::vec4(0.23f,0.22f,0.0f,1.60f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
    bransley_transform[3] = glm::mat4(glm::vec4(0.0f,0.0f,0.0f,0.0f),
                                      glm::vec4(0.0f,0.16f,0.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
}

void fill_sierpinski(){
    sierpinski_transform[0] = glm::mat4(glm::vec4(0.5f,0.00f,0.0f,0.0f),
                                      glm::vec4(0.0f,0.5f,0.0f,0.36f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
    sierpinski_transform[1] = glm::mat4(glm::vec4(0.5f,0.0f,0.0f,-0.5f),
                                      glm::vec4(0.0f,0.5f,0.0f,-0.5f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
    sierpinski_transform[2] = glm::mat4(glm::vec4(0.5f,0.0f,0.0f,0.5f),
                                      glm::vec4(0.0f,0.5f,0.0f,-0.5f),
                                      glm::vec4(0.0f,0.0f,1.0f,0.0f),
                                      glm::vec4(0.0f,0.0f,0.0f,1.0f));
}

void compute_with_shader(float *vertices, ComputeShader &computeShader){
    static int global_iteration_count = 0;

    computeShader.use();
    GLint transform_array_location = glGetUniformLocation(computeShader.ID, "u_transformations[0]");
    if(transform_array_location == -1){
        std::cout << "Warning: uniform 'u_transformations[0]' not found (location = -1)." << std::endl;
    } else {
        glUniformMatrix4fv(transform_array_location, 3, GL_TRUE, glm::value_ptr(sierpinski_transform[0]));
    }
    for(int i = 0; i < iterations; i++){
        global_iteration_count++;
        computeShader.setInt("u_seed", global_iteration_count);

        glDispatchCompute(number_of_points, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        // const int LOG_INTERVAL = 60;
        // const int N_PRINT = 10; // number of points (vec4) to print
        // static int last_logged = 0;
        // if ((global_iteration_count - last_logged) >= LOG_INTERVAL || just_transformed) {
        //     last_logged = global_iteration_count;
        //     std::vector<float> readback(N_PRINT * 4);
        //     // Ensure the SSBO (VBO) is bound to GL_SHADER_STORAGE_BUFFER
        //     glBindBuffer(GL_SHADER_STORAGE_BUFFER, VBO);
        //     // Read the first N_PRINT vec4s
        //     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * (N_PRINT * 4), readback.data());
        //     std::cout << "GPU positions (first " << N_PRINT << ") at seed " << global_iteration_count << ":\n";
        //     for (int p = 0; p < N_PRINT; ++p) {
        //         float x = readback[p * 4 + 0];
        //         float y = readback[p * 4 + 1];
        //         float z = readback[p * 4 + 2];
        //         float w = readback[p * 4 + 3];
        //         std::cout << "  [" << p << "] = (" << x << ", " << y << ", " << z << ", " << w << ")\n";
        //     }
        // }

        //glBindBuffer(GL_ARRAY_BUFFER, VBO); 
        //glBufferData(GL_ARRAY_BUFFER, sizeof(float) * number_of_vertices, vertices, GL_DYNAMIC_DRAW);
        
    }
    // shader.use();
    // glBindVertexArray(VAO);
    // glDrawArrays(GL_POINTS, 0, number_of_points);
}

void compute_bransley_shader(float *vertices,ComputeShader &computeShader){
    static int global_iteration_count = 0;

    computeShader.use();
    GLint transform_array_location = glGetUniformLocation(computeShader.ID, "u_transformations[0]");
    if(transform_array_location == -1){
        std::cout << "Warning: uniform 'u_transformations[0]' not found (location = -1)." << std::endl;
    } else {
        glUniformMatrix4fv(transform_array_location, 4, GL_TRUE, glm::value_ptr(bransley_transform[0]));
    }
    for(int i = 0; i < iterations; i++){
        global_iteration_count++;
        computeShader.setInt("u_seed", global_iteration_count);

        glDispatchCompute(number_of_points, 1, 1);
    // Ensure writes to the SSBO are visible to subsequent vertex attribute fetches.
    // When a buffer written by a shader is then used as a vertex attrib source,
    // GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT must be included.
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    }
}

void generate_points(float *vertices){
    for(unsigned int i = 0; i < number_of_vertices; i+=4){
        float V_x = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high - low)));
        float V_y = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high - low)));
        //float V_z = low + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(high - low)));
        vertices[i] = V_x;
        vertices[i + 1] = V_y;
        vertices[i + 2] = 0.0f;
        vertices[i + 3] = 1.0f;
    }
}