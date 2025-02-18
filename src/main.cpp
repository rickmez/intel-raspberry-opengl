#include <iostream>
#include <thread>
#include <memory>
#include <iostream>
#include <chrono>  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <omp.h>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <librealsense2/rs.hpp> 
#include <librealsense2/rsutil.h> 
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <pthread.h>

#include "visualization/visualization.hpp"
#include "camera/camera.hpp"
#include "ransac/ransac.hpp"
#include <map>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> // Include GLM header for matrix operations
#include <Eigen/Dense>

// // depth and rgb images
// pthread_t real_sense_thread;

// OpenCV-related variables
int width = 0, height = 0, stride = 0, depth_stride = 0;
int d_width = 0, d_height = 0, d_stride = 0;

cv::Mat frame_mat, resized_frame;

rs2::frame depthFrame;
rs2::frame color;
const rs2::vertex* vertices;

// colorframe
uint8_t color_frame = 0;
uint8_t depth_frame_flag = 0;

// bool point_cloud_available = false;
// bool point_cloud_data = false;

const void* video_data;
const void* depth_data;
uint16_t *depth;

uint8_t *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;

bool done = false;


int g_argc;
char **g_argv;

uint8_t flag_points = 0;
uint8_t config_flag = 0;

pthread_t update_thread;
pthread_t frames_thread;
pthread_t points_thread;
pthread_t opengl_thread;
pthread_t filtered_thread;
pthread_t ransac_thread;
pthread_t cleanUp_thread;

pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;

// std::mutex localPointCloudMutex;

int window;
GLuint gl_depth_tex;

// Intrinsic parameters for intel 
const double fx = 428.299;  // Focal length x
const double fy = 428.299;  // Focal length y
const double cy = 244.483;   // Principal point x
const double cx = 419.197;   // Principal point y

rs2::pointcloud pc;  // Point cloud object
rs2::points points;  // Container for calculated point cloud data
rs2::frame_queue filtered_data;
rs2::frame_queue rgbframe_data;

const int MIN_DEPTH = 0.01;  // Minimum valid depth value in millimeters
const int MAX_DEPTH = 0.75; // Maximum valid depth value in millimeters

// Camera position (for translating the camera)
float cameraX = 0.0f, cameraY = 0.0f, cameraZ = -5.0f; // Start far enough to see the points

// // Last mouse position
int lastMouseX, lastMouseY;
bool leftMouseDown = false;
bool rightMouseDown = false;  // For rotating

// Scroll sensitivity for zooming
float zoomSensitivity = 0.5f;

// IMU data
rs2_vector gyro_data;
rs2_vector acce_data;

float s_pitch = 0.0f, s_roll = 0.0f;
float gyro_x = 0.0f, gyro_y = 0.0f;
float alpha = 0.98f;
float dt = 0.01f; // Assuming 100Hz IMU update rate

uint8_t gen_thread, run_cam_thread, proc_thread = 0;

uint8_t ransac_data = 0;
PlaneModel result;


std::vector<PointCloudData> pointCloud;

// Shared data structure
struct SharedData {
    std::vector<Eigen::Vector3d> pointCloud; // Point cloud data
    std::vector<Eigen::Vector3d> inliers;    // Inliers from RANSAC
    bool pointCloudReady = false;            // Flag to indicate new point cloud is available
    std::mutex mutex;                        // Mutex for thread-safe access
};

SharedData sharedData; // Global shared data

// Global variables for arcball rotation
glm::quat rotation = glm::quat(.45, 0.55, -0.46, -0.5);  // identity quaternion
int windowWidth = 640;   // Update these with your window dimensions
int windowHeight = 480;

glm::vec3 getArcballVector(int x, int y, int width, int height) {
    glm::vec3 P(
        1.0f * x / width * 2 - 1.0f,
        1.0f * y / height * 2 - 1.0f,
        0.0f
    );
    // Invert y so that +Y is up
    P.y = -P.y;
    float OP_squared = P.x * P.x + P.y * P.y;
    if (OP_squared <= 1.0f)
        P.z = sqrt(1.0f - OP_squared);  // Pythagoras
    else
        P = glm::normalize(P);  // nearest point on sphere
    return P;
}

void reshape(int width, int height) {
    if (height == 0) height = 1;
    windowWidth = width;
    windowHeight = height;

    float aspectRatio = static_cast<float>(width) / height;
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float nearPlane = 0.01f;
    float farPlane = 10000.0f;
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspectRatio, nearPlane, farPlane);
    glLoadMatrixf(glm::value_ptr(projection));

    glMatrixMode(GL_MODELVIEW);
}

// Function to handle mouse dragging
void mouseMotion(int x, int y) {
    int deltaX = x - lastMouseX;
    int deltaY = y - lastMouseY;

    if (leftMouseDown) {
        // Translate the camera position based on mouse movement
        cameraX += deltaX * 0.02f;  // Adjust sensitivity as needed
        cameraY -= deltaY * 0.02f;  // Inverted Y-axis
        glutPostRedisplay();
    }

    if (rightMouseDown) {
        // Arcball rotation update
        glm::vec3 va = getArcballVector(lastMouseX, lastMouseY, windowWidth, windowHeight);
        glm::vec3 vb = getArcballVector(x, y, windowWidth, windowHeight);
        // Compute the angle between the two vectors
        float angle = acos(std::min(1.0f, glm::dot(va, vb)));
        // Compute the rotation axis (it should be perpendicular to both)
        glm::vec3 axis = glm::cross(va, vb);
        if (glm::length(axis) > 1e-6) {  // Ensure non-zero rotation
            axis = glm::normalize(axis);
            // Create a quaternion for this incremental rotation
            glm::quat q = glm::angleAxis(angle, axis);
            // Update the global rotation (note the order: new rotation * current rotation)
            rotation = q * rotation;
        }
        glutPostRedisplay();
    }

    // Update last mouse position
    lastMouseX = x;
    lastMouseY = y;
}
// Function to handle mouse button events
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            leftMouseDown = true;
            lastMouseX = x;
            lastMouseY = y;
        } else if (state == GLUT_UP) {
            leftMouseDown = false;
        }
    }
    
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) {
            rightMouseDown = true;
            lastMouseX = x;
            lastMouseY = y;
        } else if (state == GLUT_UP) {
            rightMouseDown = false;
        }
    }
}

// Function to handle the mouse scroll (zoom)
void mouseWheel(int button, int dir, int x, int y) {
    if (dir > 0) {
        // Scroll up - Zoom in
        cameraZ += zoomSensitivity;
    } else {
        // Scroll down - Zoom out
        cameraZ -= zoomSensitivity;
    }

    // Request a redraw
    glutPostRedisplay();
}


void drawArrowHead(float x, float y, float z, float r, float g, float b) {
    glColor3f(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);
    glutSolidCone(0.05, 0.1, 10, 10); // Puntas pequeñas
    glPopMatrix();
}

void drawXYZAxes() { 
    glLineWidth(2.0f);

    glBegin(GL_LINES);

    // Eje X en rojo
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    // Eje Y en verde
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    // Eje Z en azul
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);

    glEnd();

    // Dibujar puntas de flecha
    drawArrowHead(1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f); // X (rojo)
    drawArrowHead(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f); // Y (verde)
    drawArrowHead(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f); // Z (azul)
}


float ax, ay, az , gx, gy = 0;

uint8_t got_depth = 0;
uint8_t got_Inliers = 0;

void renderText(float x, float y, const char *text, void *font) {
    glRasterPos2f(x, y); // Position for text rendering
    while (*text) {
        glutBitmapCharacter(font, *text);
        text++;
    }
}


void drawGrid(float size = 10.0f, float step = 1.0f) {
    glColor4f(0.3f, 0.3f, 0.3f, 0.5f); // Dark gray color for the grid
    glLineWidth(1.0f);

    glBegin(GL_LINES);
    
    for (float i = -size; i <= size; i += step) {
        // Lines along X axis
        glVertex3f(i, 1.0f, -size);
        glVertex3f(i, 1.0f, size);

        // Lines along Z axis
        glVertex3f(-size, 1.0f, i);
        glVertex3f(size, 1.0f, i);
    }

    glEnd();
}


// Global buffers for point cloud double-buffering.
std::vector<PointCloudData> pointCloudBuffer[2];
int activeBufferIndex = 0; // Buffer used by the renderer.
std::atomic<bool> point_cloud_available(false);
std::atomic<bool> point_cloud_data(false);

// Global frame counter.
std::atomic<uint32_t> frame_number(0);

// Mutex to protect shared access.
std::mutex localPointCloudMutex;


void DrawGLScene() {
    pthread_mutex_lock(&gl_backbuf_mutex);
    
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Compute and load the view matrix.
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(cameraX, cameraY, cameraZ));
    view *= glm::mat4_cast(rotation);
    glLoadMatrixf(glm::value_ptr(view));

    glPointSize(4.0f);
    drawXYZAxes();
    drawGrid(10.0f, 1.0f);  

    // Always copy the current active buffer data.
    std::vector<PointCloudData> currentPointCloud;
    {
        std::lock_guard<std::mutex> lock(localPointCloudMutex);
        currentPointCloud = pointCloudBuffer[activeBufferIndex];
        // Do not clear point_cloud_available here; always use the last valid data.
    }

    // Render the point cloud using vertex arrays.
    if (!currentPointCloud.empty()) {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        // Use the address of the first element's data.
        glVertexPointer(3, GL_FLOAT, sizeof(PointCloudData), &currentPointCloud[0].position);
        glColorPointer(3, GL_FLOAT, sizeof(PointCloudData), &currentPointCloud[0].color);

        glDrawArrays(GL_POINTS, 0, currentPointCloud.size());

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }

    // --------------- DRAW TEXT IN SCREEN SPACE ---------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.0f, 0.0f, 0.0f);
    std::string quatText = "Rotation (quat): ";
    quatText += "w: " + std::to_string(rotation.w) + " ";
    quatText += "x: " + std::to_string(rotation.x) + " ";
    quatText += "y: " + std::to_string(rotation.y) + " ";
    quatText += "z: " + std::to_string(rotation.z);
    renderText(20, windowHeight - 20, quatText.c_str(), GLUT_BITMAP_HELVETICA_18);
    point_cloud_available = false;

    // Restore matrices.
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
    pthread_mutex_unlock(&gl_backbuf_mutex);
}


void keyboard(unsigned char key, int x, int y) {
    if (key == 27) { // 27 is the ASCII code for the Escape key
        // cleanup here
        done = true;
        
        pthread_join(update_thread, NULL);
        pthread_join(frames_thread, NULL);
        pthread_join(points_thread, NULL);

        std::cout<< "CleanUp complete\n"<<std::endl;
        glutDestroyWindow(glutGetWindow());
    }
}

void* RealSenseThread(void* arg) {
    // Register the cleanup function
    // atexit(cleanup);
    
    printf("GL thread\n");
    glutInit(&g_argc, g_argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);
    window = glutCreateWindow("Realsense Point Cloud");
    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene);
	glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMouseWheelFunc(mouseWheel);  // Register the mouse scroll function
    glutMotionFunc(mouseMotion);
    glutKeyboardFunc(keyboard);
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0);
    glutMainLoop();


    return NULL;
}


void InitGL()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set to white background
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_depth_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // ReSizeGLScene(640, 480);
}

// void* RANSAC_INLIERS(void* arg){
//     // pthread_mutex_lock(&gl_backbuf_mutex);
//     while(!done) {
        
//         std::lock_guard<std::mutex> lock(localPointCloudMutex);
//         if(ransac_data){
//             // std::cout << "Number of inliers: " << localPointCloud.size() << std::endl;
//             result = fit_plane_ransac(pointCloud, 100, 100, Eigen::Vector3d(1, 0, 0));
//             // std::cout<<result.inliers.size()<<std::endl;
//             ransac_data = 0;
//             got_Inliers = 1;
//         }
//         // // Check if a new point cloud is available
//         // {
//         //     std::lock_guard<std::mutex> lock(sharedData.mutex);
//         //     if (sharedData.pointCloudReady) {
//         //         localPointCloud = sharedData.pointCloud; // Copy the point cloud
//         //         sharedData.pointCloudReady = false;      // Reset the flag
//         //     }
//         // }

//         // // Run RANSAC if a new point cloud is available
//         // if (!localPointCloud.empty()) {

//         //     // Save the inliers
//         //     {
//         //         std::lock_guard<std::mutex> lock(sharedData.mutex);
//         //         sharedData.inliers = result.inliers;
//         //     }

//         //     // Optionally, print or save the inliers
//         // }
        
//     }
//     std::cout<< "closing thread: RANSAC_INLIERS "<<std::endl;
//     pthread_exit(NULL);
//     return NULL;
// }

// uint32_t frame_number = 0;

void* pointcloud_generate(void* arg) {
    const float scaleX = 1280.0f / 848.0f;
    const float scaleY = 720.0f / 480.0f;
    const int step_x = 1;
    const int step_y = 1;

    while (!done) {
        if (point_cloud_data & config_flag) {
            std::lock_guard<std::mutex> lock(localPointCloudMutex);
            int writeBuffer = 1 - activeBufferIndex;
            pointCloudBuffer[writeBuffer].clear();
            
            for (int y = 0; y < 480; y += step_y) {
                for (int x = 0; x < 848; x += step_x) {
                    int i = y * 848 + x;
                    int depth_value = (depth_front[3 * i + 0] << 8) | depth_front[3 * i + 1];

                    if (depth_value > 0 && depth_value < 4000) {
                        float z = depth_value * 0.001f;
                        float x_world = (x - cx) * z / fx;
                        float y_world = (y - cy) * z / fy;
                        
                        Eigen::Vector3f rotated(x_world, y_world, z);

                        int rgb_x = static_cast<int>(x * scaleX);
                        int rgb_y = static_cast<int>(y * scaleY);
                        int rgb_idx = (rgb_y * 1280 + rgb_x) * 3;

                        Eigen::Vector3f color(
                            rgb_front[rgb_idx + 0] / 255.0f,
                            rgb_front[rgb_idx + 1] / 255.0f,
                            rgb_front[rgb_idx + 2] / 255.0f
                        );

                        pointCloudBuffer[writeBuffer].emplace_back(
                            rotated.x(), rotated.y(), rotated.z(),
                            color.x(), color.y(), color.z()
                        );
                    }
                }
            }
            
            frame_number++;
            if (frame_number % 10 == 0) {
                std::cout << "\rFrame number: " << frame_number
                          << " | Point cloud size: " 
                          << pointCloudBuffer[writeBuffer].size()
                          << "     " << std::flush;
            }
            // Swap buffers atomically.
            activeBufferIndex = writeBuffer;
            point_cloud_available = true;
            point_cloud_data = false;
        }
    }
    
    std::cout << "closing thread: pointcloud_generate" << std::endl;
    pthread_exit(NULL);
    return NULL;
}


    // while (!done) {
        //     if(done) break;
    //     if (point_cloud_data & config_flag) {

    //         std::lock_guard<std::mutex> lock(localPointCloudMutex);
    //         pointCloud.clear();
    //         // localPointCloud.clear();

    //         for (int y = 0; y < 480; y += step_y) {
    //             for (int x = 0; x < 848; x += step_x) {
    //                 int i = y * 848 + x;
    //                 int depth_value = depth_front[3 * i + 0] << 8 | depth_front[3 * i + 1];

    //                 if (depth_value > 0 && depth_value < 4000) {
    //                     float z = depth_value * 0.001f;
    //                     float x_world = (x - cx) * z / fx;
    //                     float y_world = (y - cy) * z / fy;
    //                     // std::cout<< x_world << std::endl;
    //                     // float cos_theta = cos(s_pitch);
    //                     // float sin_theta = sin(s_pitch);
    //                     // float cos_alpha = cos(s_roll);
    //                     // float sin_alpha = sin(s_roll);

    //                     Eigen::Vector3f rotated(
    //                         x_world,// cos_theta * x_world - sin_alpha * sin_theta * y_world + cos_alpha * sin_theta * z,
    //                         y_world,// cos_alpha * y_world + sin_alpha * z,
    //                         z// -sin_theta * x_world - sin_alpha * cos_theta * y_world + cos_alpha * cos_theta * z
    //                     );

    //                     // Map depth pixel to corresponding RGB pixel
    //                     int rgb_x = static_cast<int>(x * scaleX);
    //                     int rgb_y = static_cast<int>(y * scaleY);
    //                     int rgb_idx = (rgb_y * 1280 + rgb_x) * 3; // RGB buffer index


    //                     Eigen::Vector3f color(
    //                         rgb_front[rgb_idx + 0] / 255.0f,
    //                         rgb_front[rgb_idx + 1] / 255.0f,
    //                         rgb_front[rgb_idx + 2] / 255.0f
    //                     );

    //                     // std::cout<< "3d points" << rotated.x()<< " " <<  rotated.y() << " " << rotated.z()<<std::endl;
    //                     pointCloud.emplace_back(rotated.x(), rotated.y(), rotated.z(), color.x(), color.y(), color.z());
    //                     // localPointCloud.push_back(rotated);
    //                 }
    //             }
    //         }
    //         frame_number++;
    //         if(frame_number % 10 == 0){
    //             std::cout << "\rFrame number: " << frame_number
    //               << " | Point cloud size: " << pointCloud.size()
    //               << "     " // Extra spaces to overwrite old characters
    //               << std::flush;
    //         }
    //         ransac_data++;
    //         point_cloud_available = true;
    //         point_cloud_data = false;
    //     }
    // }


uint8_t got_rgb = 0;

void* process_frame_raw_data(void* arg) {

    while(!done){
        if(done) break;
        if (depth_frame_flag && color_frame && config_flag) {
            // pthread_mutex_lock(&gl_backbuf_mutex);
            

            uint16_t *depth = (uint16_t *)depth_data;
            
            if(rgb_back != (uint8_t*)video_data){
                rgb_back = (uint8_t*)video_data;
                rgb_mid = rgb_back;
                got_rgb = 1;
            }

            if(got_rgb){
                rgb_front = rgb_mid;
                got_rgb = 0;
            }

            for (int i = 0; i < 848 * 480; i++) {
                int pval = depth[i];
                depth_mid[3 * i + 0] = (pval >> 8) & 0xff;  // High byte (Red)
                depth_mid[3 * i + 1] = pval & 0xff;         // Low byte (Green)
                depth_mid[3 * i + 2] = 0;                   // Unused for now

            }

            // ax = acce_data.x;
            // ay = acce_data.y;
            // az = acce_data.z;
            
            // gx = gyro_data.x;
            // gy = gyro_data.y;

            // float accel_pitch = atan2(sqrt(ax * ax + ay * ay), az);
            // float accel_roll = atan2(ay, ax);

            // gyro_x += gx * dt;
            // gyro_y += gy * dt;

            // s_pitch = alpha * (s_pitch + gx * dt) + (1 - alpha) * accel_pitch;
            // s_roll = alpha * (s_roll + gy * dt) + (1 - alpha) * accel_roll;

            // std::cout<< s_pitch << " " << s_roll << std::endl;
            
            got_depth = 1;
            
            if (got_depth) {
                depth_front = depth_mid;
                got_depth = 0;
                point_cloud_data = true;
            }
 
            if (!depth_front || !rgb_front) {
                std::cerr << "Error: Null pointer detected" << std::endl;
                pthread_exit(NULL);
            }
 
            color_frame = 0; // Reset the flag
            depth_frame_flag = 0; // Reset the flag
            // pthread_mutex_unlock(&gl_backbuf_mutex);
        }
    }
    proc_thread++;
    std::cout<< "closing thread: process_frame_raw_data "<<std::endl;
    pthread_exit(NULL);
    return NULL;
}


void* process_filtered(void *arg){
    while(!done){

        if(done) break;
        rs2::frame f;
        
        if(filtered_data.poll_for_frame(&f)){
            
            // motion frames
            // try{
            //     gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
            //     acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
            //     // auto pose = (frameset.get_pose_frame()).as<rs2::pose_frame>();
            // }
            // catch (const rs2::error& e) {
            //     std::cerr << "RealSense error: " << e.what() << std::endl;
            // }

            // color = frameset.get_color_frame();
            
            if (!f) {
                std::cerr << "Invalid frames!" << std::endl;
            }
            else{
                
                auto depth_f = f.as<rs2::depth_frame>(); 

                depth_data = depthFrame.get_data();
        
            }
        }

        rs2::frame rgbf;

        if(rgbframe_data.poll_for_frame(&rgbf)){
            
            // motion frames
            // try{
            //     gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
            //     acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
            //     // auto pose = (frameset.get_pose_frame()).as<rs2::pose_frame>();
            // }
            // catch (const rs2::error& e) {
            //     std::cerr << "RealSense error: " << e.what() << std::endl;
            // }

            // color = frameset.get_color_frame();
            
            if (!rgbf) {
                std::cerr << "Invalid frames!" << std::endl;
            }
            else{
                auto videFrame = rgbf.as<rs2::video_frame>(); 
                width = videFrame.get_width();
                height = videFrame.get_height();   
                stride = videFrame.get_stride_in_bytes();
                video_data = videFrame.get_data();                    
            }
            if (!depth_data || !video_data) 
                std::cerr << "Invalid frames!" << std::endl;
            else{
                color_frame = 1;
                depth_frame_flag = 1;
            }
        }
    }  
    std::cout << "closing thread: process_filtered\n" << std::endl;
    pthread_exit(NULL);
    return NULL;
} 


void* run_camera(void *arg){
    try {
        
        std::string bag_file = "/home/rick/Downloads/20250115_153938.bag";
        auto pipe_video = std::make_shared<rs2::pipeline>();
        
        rs2::config cfg_video;
        
        // enable different modes
        cfg_video.enable_device_from_file(bag_file);      // from a file
        cfg_video.enable_stream(RS2_STREAM_DEPTH);
        cfg_video.enable_stream(RS2_STREAM_COLOR);
        cfg_video.enable_stream(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
        cfg_video.enable_stream(RS2_STREAM_ACCEL,RS2_FORMAT_MOTION_XYZ32F);
        // cfg_sensor.enable_stream(RS2_STREAM_POSE,RS2_FORMAT_MOTION_XYZ32F);
        
        // Declare filters
        rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
        // rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
        rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
        rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise


        rs2::frameset frameset;
        rs2::frame gyro_frame;
        rs2::frame acce_frame;

        // ########## THESE FUNCTIONS ONLY NEED TO BE RUN ONCE TO EXTRACT THE INTRINSICS
        // auto Depth_sensor = get_a_sensor_from_a_device(device);
        // auto Depth_stream_profile = choose_a_streaming_profile(Depth_sensor);
        // get_field_of_view(Depth_stream_profile);

        pipe_video->start(cfg_video);
        
        auto device = pipe_video->get_active_profile().get_device();
        
        if(device.as<rs2::playback>()){
            
            rs2::playback playback = device.as<rs2::playback>();
            // playback.set_real_time(false);

            uint64_t posCurr = playback.get_position();
            
            config_flag = 1;
            // pipe->try_wait_for_frames(&frameset, 1000)
            while (pipe_video->try_wait_for_frames(&frameset, 5000) && !done) {
                
                // #### Alternate ways to get a frameset ####
                // auto frameset = pipe_video->wait_for_frames(1000);
                // auto frameset_sensor = pipe_sensor->wait_for_frames(1000);
                
                // motion frames
                // try{
                //     gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
                //     acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
                //     // auto pose = (frameset.get_pose_frame()).as<rs2::pose_frame>();
                // }
                // catch (const rs2::error& e) {
                //     std::cerr << "RealSense error: " << e.what() << std::endl;
                // }
                
                // depth frames
                depthFrame = (frameset.get_depth_frame()).as<rs2::depth_frame>();
                // rgb frames
                color = frameset.get_color_frame();
                


                if (!depthFrame) {
                    std::cerr << "Invalid frames in run camera!" << std::endl;
                }
                else{
                    
                    // get motion
                    // auto pose_data = pose.get_pose_data();
                    // auto gyro_motion = gyro_frame.as<rs2::motion_frame>();
                    // gyro_data = gyro_motion.get_motion_data();
                    
                    // auto acce_motion = acce_frame.as<rs2::motion_frame>();
                    // acce_data = acce_motion.get_motion_data();
                    // get color
                    
                    // auto videoframe = color.as<rs2::video_frame>();
                    // width = videoframe.get_width();
                    // height = videoframe.get_height();
                    
                    // stride = videoframe.get_stride_in_bytes();
                    // video_data = videoframe.get_data();
                    
                    rs2::frame filtered_dec = dec_filter.process(depthFrame);
                    rs2::frame filtered_spa = spat_filter.process(filtered_dec);
                    rs2::frame filtered_tem = temp_filter.process(depthFrame);

                    filtered_data.enqueue(filtered_tem);
                    rgbframe_data.enqueue(color);
                    // depth_data = depthFrame.get_data();
                    

                    // if (!video_data || !depth_data) 
                    //     std::cerr << "Invalid frames!" << std::endl;
                    // else{
                    //     color_frame = 1;
                    //     depth_frame_flag = 1;
                    // }
            
                }
                


                auto posNext = playback.get_position();
                if (posNext < posCurr){
                    std::cout<< "\nDone with playback\n"<<std::endl;
                    done = true;
                }

                if(done) break;
                
                posCurr = posNext;
                
                

                // if (kbhit()) {
                //     char ch = getchar();
                //     if (ch == 27) {  // 27 is the ASCII code for ESC
                //         point_cloud_available = false;
                //         std::cout << "ESC pressed. Exiting...\n";
                //         break;
                //     }
                // }


                // auto posNext_s = playback_sensor.get_position();
                // if (posNext < posCurr_s) break;
                // posCurr_s = posNext_s;
                
            }
        }
        


        done = true;
        pipe_video->stop();
        std::cout<< "closing thread: run_camera"<<std::endl;
        pthread_exit(NULL);
    }


    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return NULL;
}


int main(int argc, char* argv[]) {
    
    // CreateAndInitWindow();
    std::cout << "Intel RealSense SDK Version: " << RS2_API_VERSION_STR << std::endl;


    depth_mid = (uint8_t *)malloc(848 * 480 * 3);
    depth_front = (uint8_t *)malloc(848 * 480 * 3);
    
    rgb_back = (uint8_t*)malloc(1280 * 720 * 3);
    rgb_mid = (uint8_t*)malloc(1280 * 720 * 3);
	rgb_front = (uint8_t*)malloc(1280 * 720 * 3);

    g_argc = argc;
    g_argv = argv;

    // Create the RealSense thread
    if (pthread_create(&update_thread, NULL, run_camera, NULL) != 0) {
        std::cerr << "Error creating run_camera thread!" << std::endl;
        return -1;
    }

    if (pthread_create(&filtered_thread, NULL, process_filtered, NULL) != 0) {
        std::cerr << "Error creating process_filtered thread!" << std::endl;
        return -1;
    }
    
    // Create the second thread
    if (pthread_create(&frames_thread, NULL, process_frame_raw_data, NULL) != 0) {
        std::cerr << "Error creating process_frame_raw_data thread!" << std::endl;
        return -1;
    }

    // Create the second thread
    if (pthread_create(&points_thread, NULL, pointcloud_generate, NULL) != 0) {
        std::cerr << "Error creating depth_stream thread!" << std::endl;
        return -1;
    }

    // Create the second thread
    if (pthread_create(&opengl_thread, NULL, RealSenseThread, NULL) != 0) {
        std::cerr << "Error creating RealSenseThread thread!" << std::endl;
        return -1;
    }

    // // Create the second thread
    // if (pthread_create(&ransac_thread, NULL, RANSAC_INLIERS, NULL) != 0) {
    //     std::cerr << "Error creating RANSAC_INLIERS thread!" << std::endl;
    //     return -1;
    // }

    // Wait for threads to finish
    pthread_join(update_thread, NULL);
    pthread_join(frames_thread, NULL);
    pthread_join(points_thread, NULL);
    pthread_join(opengl_thread, NULL);
    // pthread_join(ransac_thread, NULL);

    // run_camera();

    return 0;
}
