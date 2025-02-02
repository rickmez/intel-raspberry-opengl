#include <iostream>
#include <thread>
#include <memory>
#include <iostream>
#include <chrono>  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <thread>
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
#include <map>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> // Include GLM header for matrix operations

// // depth and rgb images
// pthread_t real_sense_thread;

// OpenCV-related variables
int width = 0, height = 0, stride = 0, depth_stride = 0;
int d_width = 0, d_height = 0, d_stride = 0;

cv::Mat frame_mat, resized_frame;

rs2::frame depthFrame;
rs2::frame color;
const rs2::vertex* vertices;

size_t num_points = 0;

// colorframe
uint8_t color_frame = 0;
uint8_t depth_frame_flag = 0;

bool point_cloud_available = false;
bool point_cloud_data = false;

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

pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;

int window;
GLuint gl_depth_tex;

// Intrinsic parameters for intel 
const double fx = 428.299;  // Focal length x
const double fy = 428.299;  // Focal length y
const double cy = 244.483;   // Principal point x
const double cx = 419.197;   // Principal point y

rs2::pointcloud pc;  // Point cloud object
rs2::points points;  // Container for calculated point cloud data

const int MIN_DEPTH = 0.01;  // Minimum valid depth value in millimeters
const int MAX_DEPTH = 0.75; // Maximum valid depth value in millimeters
// Camera rotation angles (for rotating the scene)
float pitch = -179.0f;  // Rotation around x-axis
float yaw = -29.0f;    // Rotation around y-axis

// Camera position (for translating the camera)
float cameraX = 5.0f, cameraY = -5.0f, cameraZ = -13.0f; // Start far enough to see the points

// // Last mouse position
int lastMouseX, lastMouseY;
bool leftMouseDown = false;
bool rightMouseDown = false;  // For rotating

// Scroll sensitivity for zooming
float zoomSensitivity = 1.0f;

// IMU data
rs2_vector gyro_data;
rs2_vector acce_data;

float s_pitch = 0.0f, s_roll = 0.0f;
float gyro_x = 0.0f, gyro_y = 0.0f;
float alpha = 0.98f;
float dt = 0.01f; // Assuming 100Hz IMU update rate


struct PointCloudData {
    float x;   // X coordinate
    float y;   // Y coordinate
    float z;   // Z coordinate
    float r;   // Red
    float g;   // Green
    float b;   // Blue

    // Optional constructor for convenience
    PointCloudData(float x_, float y_, float z_, float r_ = 1.0f, float g_ = 1.0f, float b_ = 1.0f)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
};

std::vector<PointCloudData> pointCloud;

// Function to handle window resizing
void reshape(int width, int height) {
    if (height == 0) height = 1;

    float aspectRatio = (float)width / (float)height;

    glViewport(0, 0, width, height);
    
    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // Adjust the near and far clipping planes
    float nearPlane = 0.01f;  // Adjust as necessary
    float farPlane = 10000.0f;  // Adjust far plane based on your scene

    // Set perspective projection using GLM
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
        cameraX += deltaX * 0.5f;  // Adjust sensitivity with the factor
        cameraY -= deltaY * 0.5f;  // Inverted Y-axis movement

        // Request a redraw
        glutPostRedisplay();
    }

    if (rightMouseDown) {
        // Rotate the camera (yaw and pitch) when the right button is held
        yaw += deltaX * 4.2f;    // Adjust 0.2f for sensitivity
        pitch += deltaY * 4.2f;

        // Clamp pitch to prevent flipping
        if (pitch > 179.0f) pitch = 179.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        glutPostRedisplay();
    }

    // Update the last mouse position
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

void drawXYZAxes() {
    glLineWidth(2.0f); // Ajusta el grosor de las líneas

    glBegin(GL_LINES);

    // Eje X en rojo
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1000.0f, 0.0f, 0.0f);
    glVertex3f(1000.0f, 0.0f, 0.0f);

    // Eje Y en verde
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, -1000.0f, 0.0f);
    glVertex3f(0.0f, 1000.0f, 0.0f);

    // Eje Z en azul
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, -1000.0f);
    glVertex3f(0.0f, 0.0f, 1000.0f);

    glEnd();
}

float ax, ay, az , gx, gy = 0;

uint8_t got_depth = 0;

void DrawGLScene()
{
    pthread_mutex_lock(&gl_backbuf_mutex);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Apply camera transformations
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(cameraX, cameraY, cameraZ));
    view = glm::rotate(view, glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
    glLoadMatrixf(glm::value_ptr(view));

    glPointSize(3.0f);
    drawXYZAxes();

    if (point_cloud_available) {
        glBegin(GL_POINTS);

        for (const auto& point : pointCloud) {
            glColor3f(point.r, point.g, point.b);
            glVertex3f(point.x, point.y, point.z);
            // std::cout << "Position: (" << point.x << ", " << point.y << ", " << point.z << ")\n";
            // std::cout << "Color: R=" << point.r << ", G=" << point.g << ", B=" << point.b << "\n";
        }
        
        glEnd();
    }
    
    pthread_mutex_unlock(&gl_backbuf_mutex);

    glutSwapBuffers();
}



void* pointcloud_generate(void* arg){
    const float scaleX = 1280.0f / 848.0f;
    const float scaleY = 720.0f / 480.0f;
    while(!done){
        if(point_cloud_data){
            
            pointCloud.clear();

            for (int y = 0; y < 480; ++y) {
                for (int x = 0; x < 848; ++x) {
                    int i = y * 848 + x;
                    
                    // Extract depth value (convert 16-bit depth to mm)
                    int depth_value = depth_front[3 * i + 0] << 8 | depth_front[3 * i + 1];

                    if (depth_value > 0 && depth_value < 5000) { // Valid depth range (0-5 meters)
                        // Convert depth pixel to world coordinates
                        float z = depth_value * 0.001f; // Convert mm to meters
                        float x_world = (x - cx) * z / fx;
                        float y_world = (y - cy) * z / fy;

                        float cos_theta = cos(s_pitch);
                        float sin_theta = sin(s_pitch);
                        float cos_alpha = cos(s_roll);
                        float sin_alpha = sin(s_roll);

                        float x_new = cos_theta * x_world - sin_alpha * sin_theta * y_world + cos_alpha * sin_theta * z;
                        float y_new = cos_alpha * y_world + sin_alpha * z;
                        float z_new = -sin_theta * x_world - sin_alpha * cos_theta * y_world + cos_alpha * cos_theta * z;

                        // Map depth pixel to corresponding RGB pixel
                        int rgb_x = static_cast<int>(x * scaleX);
                        int rgb_y = static_cast<int>(y * scaleY);
                        int rgb_idx = (rgb_y * 1280 + rgb_x) * 3; // RGB buffer index

                        // Extract RGB color (normalize to [0,1] for OpenGL)
                        float r = rgb_front[rgb_idx + 0] / 255.0f;
                        float g = rgb_front[rgb_idx + 1] / 255.0f;
                        float b = rgb_front[rgb_idx + 2] / 255.0f;

                        // pointCloud.push_back(PointCloudData(x_world, y_world, z, r, g, b)); // Red
                        pointCloud.push_back(PointCloudData(x_new, y_new, z_new, r, g, b)); // Red

                        // glColor3f(r, g, b);
                        // glVertex3f(x_world, y_world, z);
                    }
                }
            }
            
            point_cloud_available = true;
        }
    }

    return NULL;
}

void* process_frame_raw_data(void* arg) {
    while(!done){
        if (depth_frame_flag && color_frame) {
            pthread_mutex_lock(&gl_backbuf_mutex);
            uint16_t *depth = (uint16_t *)depth_data;
            rgb_mid = (uint8_t*)video_data;
            
            uint8_t *tmp = rgb_front;
            rgb_front = rgb_mid;
            rgb_mid = tmp;
            

            for (int i = 0; i < 848 * 480; i++) {
                int pval = depth[i];
                depth_mid[3 * i + 0] = (pval >> 8) & 0xff;  // High byte (Red)
                depth_mid[3 * i + 1] = pval & 0xff;         // Low byte (Green)
                depth_mid[3 * i + 2] = 0;                   // Unused for now

            }

            ax = acce_data.x;
            ay = acce_data.y;
            az = acce_data.z;
            
            gx = gyro_data.x;
            gy = gyro_data.y;

            float accel_pitch = atan2(sqrt(ax * ax + ay * ay), az);
            float accel_roll = atan2(ay, ax);

            gyro_x += gx * dt;
            gyro_y += gy * dt;

            s_pitch = alpha * (s_pitch + gx * dt) + (1 - alpha) * accel_pitch;
            s_roll = alpha * (s_roll + gy * dt) + (1 - alpha) * accel_roll;

            // std::cout<< s_pitch << " " << s_roll << std::endl;
            
            got_depth = 1;
            
            if (got_depth) {
                uint8_t *tmp = depth_front;
                depth_front = depth_mid;
                depth_mid = tmp;
                got_depth = 0;
                point_cloud_data = true;
            }

            color_frame = 0; // Reset the flag
            depth_frame_flag = 0; // Reset the flag
            pthread_mutex_unlock(&gl_backbuf_mutex);
        }
    }
    return NULL;
}



void* RealSenseThread(void* arg) {
    
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



void* run_camera(void *arg){
    try {
        
        std::string bag_file = "/home/rick/Downloads/20250115_154457.bag";
        auto pipe_video = std::make_shared<rs2::pipeline>();
        // auto pipe_sensor = std::make_shared<rs2::pipeline>();
        
        rs2::config cfg_video;
        // rs2::config cfg_sensor;
        
        // enable different modes
        cfg_video.enable_device_from_file(bag_file);      // from a file
        cfg_video.enable_stream(RS2_STREAM_DEPTH);
        cfg_video.enable_stream(RS2_STREAM_COLOR);

        cfg_video.enable_stream(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
        cfg_video.enable_stream(RS2_STREAM_ACCEL,RS2_FORMAT_MOTION_XYZ32F);
        // cfg_sensor.enable_stream(RS2_STREAM_POSE,RS2_FORMAT_MOTION_XYZ32F);
        
        pipe_video->start(cfg_video);
        // pipe_sensor->start(cfg_sensor);
        
        auto device = pipe_video->get_active_profile().get_device();
        rs2::playback playback = device.as<rs2::playback>();
        playback.set_real_time(false);

        // auto device_sensor = pipe_sensor->get_active_profile().get_device();
        // rs2::playback playback_sensor = device_sensor.as<rs2::playback>();
        // playback_sensor.set_real_time(false);

        // playback.set_playback_speed(10.0);
        // std::vector<rs2::sensor> sensors = device.query_sensors();

        // auto Depth_sensor = get_a_sensor_from_a_device(device);
        // auto Depth_stream_profile = choose_a_streaming_profile(Depth_sensor);

        // get_field_of_view(Depth_stream_profile);

        rs2::frameset frameset;
        // rs2::pointcloud pc;  // Point cloud object
        // rs2::points points;  // Container for calculated point cloud data
        
        // const rs2::vertex* vertices; 
        size_t num_points;
        uint64_t posCurr = playback.get_position();

        // uint64_t posCurr_s = playback_sensor.get_position();
        // int indx = 0;
        
        config_flag = 1;

        // pipe->try_wait_for_frames(&frameset, 1000)
        while (pipe_video->try_wait_for_frames(&frameset, 1000)) {
            
            // auto frameset = pipe_video->wait_for_frames(1000);
            // auto frameset_sensor = pipe_sensor->wait_for_frames(1000);
            
            auto gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
            auto acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
            
            
            depthFrame = (frameset.get_depth_frame()).as<rs2::depth_frame>();
            color = frameset.get_color_frame();
            // auto pose = (frameset.get_pose_frame()).as<rs2::pose_frame>();
            if (!depthFrame || !color) {
                std::cerr << "Invalid frames!" << std::endl;
            }
            else{
                
                // get motion
                // auto pose_data = pose.get_pose_data();
                auto gyro_motion = gyro_frame.as<rs2::motion_frame>();
                gyro_data = gyro_motion.get_motion_data();
                
                auto acce_motion = acce_frame.as<rs2::motion_frame>();
                acce_data = acce_motion.get_motion_data();
                // get color
                
                auto videoframe = color.as<rs2::video_frame>();
                // width = videoframe.get_width();
                // height = videoframe.get_height();
                stride = videoframe.get_stride_in_bytes();
                video_data = videoframe.get_data();
                color_frame = 1;
                
                depth_data = depthFrame.get_data();
                depth_frame_flag = 1;

        
            }
            
            auto posNext = playback.get_position();
            if (posNext < posCurr) break;
            posCurr = posNext;

            std::string str = "ESC";
            char ch;
            if ((ch = std::cin.get()) == 27) {
                std::cout << str;   
            }


            // auto posNext_s = playback_sensor.get_position();
            // if (posNext < posCurr_s) break;
            // posCurr_s = posNext_s;
            
        }
        done = true;
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

    depth_mid = (uint8_t *)malloc(848 * 480 * 3);
    depth_front = (uint8_t *)malloc(848 * 480 * 3);
    rgb_back = (uint8_t*)malloc(848*480*3);
    rgb_mid = (uint8_t*)malloc(848*480*3);
	rgb_front = (uint8_t*)malloc(848*480*3);

    g_argc = argc;
    g_argv = argv;

    // Create the RealSense thread
    if (pthread_create(&update_thread, NULL, run_camera, NULL) != 0) {
        std::cerr << "Error creating run_camera thread!" << std::endl;
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

    // Wait for threads to finish
    pthread_join(update_thread, NULL);
    pthread_join(frames_thread, NULL);
    pthread_join(points_thread, NULL);
    pthread_join(opengl_thread, NULL);

    // run_camera();

    return 0;
}
