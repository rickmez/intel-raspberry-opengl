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
pthread_t ransac_thread;
pthread_t cleanUp_thread;

pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;

std::mutex localPointCloudMutex;

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

uint8_t gen_thread, run_cam_thread, proc_thread = 0;

uint8_t ransac_data = 0;
// std::vector<Eigen::Vector3d> localPointCloud; // Local point cloud


// struct PointCloudData {
//     float x;   // X coordinate
//     float y;   // Y coordinate
//     float z;   // Z coordinate
//     float r;   // Red
//     float g;   // Green
//     float b;   // Blue

//     // Optional constructor for convenience
//     PointCloudData(float x_, float y_, float z_, float r_ = 1.0f, float g_ = 1.0f, float b_ = 1.0f)
//         : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
// };

std::vector<PointCloudData> pointCloud;

// Shared data structure
struct SharedData {
    std::vector<Eigen::Vector3d> pointCloud; // Point cloud data
    std::vector<Eigen::Vector3d> inliers;    // Inliers from RANSAC
    bool pointCloudReady = false;            // Flag to indicate new point cloud is available
    std::mutex mutex;                        // Mutex for thread-safe access
};

SharedData sharedData; // Global shared data

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
uint8_t got_Inliers = 0;

void DrawGLScene()
{
    
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

    glBegin(GL_POINTS);
    if (point_cloud_available) {

        
        for (const auto& point : pointCloud) {
            glColor3f(point.color.x(), point.color.y(), point.color.z());
            glVertex3f(point.position.x(), point.position.y(), point.position.z());
        }


        // if(got_Inliers){

        // }

        // point_cloud_available = false;
        
    }
    glEnd();
    
    glutSwapBuffers();
    
    // pthread_mutex_unlock(&gl_backbuf_mutex);
}


void* RANSAC_INLIERS(void* arg){
    // pthread_mutex_lock(&gl_backbuf_mutex);
    while(!done) {
        
        std::lock_guard<std::mutex> lock(localPointCloudMutex);
        if(ransac_data){
            // std::cout << "Number of inliers: " << localPointCloud.size() << std::endl;
            PlaneModel result = fit_plane_ransac(pointCloud, 10, 100, Eigen::Vector3d(0, 1, 0));
            
            ransac_data = 0;
            got_Inliers = 1;
        }
        // // Check if a new point cloud is available
        // {
        //     std::lock_guard<std::mutex> lock(sharedData.mutex);
        //     if (sharedData.pointCloudReady) {
        //         localPointCloud = sharedData.pointCloud; // Copy the point cloud
        //         sharedData.pointCloudReady = false;      // Reset the flag
        //     }
        // }

        // // Run RANSAC if a new point cloud is available
        // if (!localPointCloud.empty()) {

        //     // Save the inliers
        //     {
        //         std::lock_guard<std::mutex> lock(sharedData.mutex);
        //         sharedData.inliers = result.inliers;
        //     }

        //     // Optionally, print or save the inliers
        // }
        
    }
    std::cout<< "closing thread: RANSAC_INLIERS "<<std::endl;
    pthread_exit(NULL);
    return NULL;
}


void* pointcloud_generate(void* arg) {
    const float scaleX = 1280.0f / 848.0f;
    const float scaleY = 720.0f / 480.0f;
    const int step_x = 11;
    const int step_y = 11;

    while (!done) {
        if (point_cloud_data & config_flag) {

            std::lock_guard<std::mutex> lock(localPointCloudMutex);
            pointCloud.clear();
            // localPointCloud.clear();

            for (int y = 0; y < 480; y += step_y) {
                for (int x = 0; x < 848; x += step_x) {
                    int i = y * 848 + x;
                    int depth_value = depth_front[3 * i + 0] << 8 | depth_front[3 * i + 1];

                    if (depth_value > 0 && depth_value < 4000) {
                        float z = depth_value * 0.001f;
                        float x_world = (x - cx) * z / fx;
                        float y_world = (y - cy) * z / fy;

                        float cos_theta = cos(s_pitch);
                        float sin_theta = sin(s_pitch);
                        float cos_alpha = cos(s_roll);
                        float sin_alpha = sin(s_roll);

                        Eigen::Vector3f rotated(
                            cos_theta * x_world - sin_alpha * sin_theta * y_world + cos_alpha * sin_theta * z,
                            cos_alpha * y_world + sin_alpha * z,
                            -sin_theta * x_world - sin_alpha * cos_theta * y_world + cos_alpha * cos_theta * z
                        );

                        int rgb_x = static_cast<int>(x * scaleX);
                        int rgb_y = static_cast<int>(y * scaleY);
                        int rgb_idx = (rgb_y * 1280 + rgb_x) * 3;

                        Eigen::Vector3f color(
                            rgb_front[rgb_idx + 0] / 255.0f,
                            rgb_front[rgb_idx + 1] / 255.0f,
                            rgb_front[rgb_idx + 2] / 255.0f
                        );

                        pointCloud.emplace_back(rotated.x(), rotated.y(), rotated.z(), color.x(), color.y(), color.z());
                        // localPointCloud.push_back(rotated);
                        ransac_data++;
                    }
                }
            }
            point_cloud_available = true;
        }
    }
    std::cout << "closing thread: pointcloud_generate" << std::endl;
    pthread_exit(NULL);
    return NULL;
}


uint8_t got_rgb = 0;

void* process_frame_raw_data(void* arg) {

    while(!done){
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
                // uint8_t *tmp = depth_front;
                depth_front = depth_mid;
                // depth_mid = tmp;
                got_depth = 0;
                point_cloud_data = true;
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
        
        pipe_video->start(cfg_video);
        
        auto device = pipe_video->get_active_profile().get_device();
        rs2::playback playback = device.as<rs2::playback>();
        playback.set_real_time(false);

        // ########## THESE FUNCTIONS ONLY NEED TO BE RUN ONCE TO EXTRACT THE INTRINSICS
        // auto Depth_sensor = get_a_sensor_from_a_device(device);
        // auto Depth_stream_profile = choose_a_streaming_profile(Depth_sensor);
        // get_field_of_view(Depth_stream_profile);


        rs2::frameset frameset;
        rs2::frame gyro_frame;
        rs2::frame acce_frame;

        uint64_t posCurr = playback.get_position();

        config_flag = 1;

        // pipe->try_wait_for_frames(&frameset, 1000)
        while (pipe_video->try_wait_for_frames(&frameset, 1000) && !done) {
            
            // #### Alternate ways to get a frameset ####
            // auto frameset = pipe_video->wait_for_frames(1000);
            // auto frameset_sensor = pipe_sensor->wait_for_frames(1000);
            
            // motion frames
            try{
                gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
                acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
                // auto pose = (frameset.get_pose_frame()).as<rs2::pose_frame>();
            }
            catch (const rs2::error& e) {
                std::cerr << "RealSense error: " << e.what() << std::endl;
            }
            
            // depth frames
            depthFrame = (frameset.get_depth_frame()).as<rs2::depth_frame>();
            // rgb frames
            color = frameset.get_color_frame();
            


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
                width = videoframe.get_width();
                height = videoframe.get_height();
                
                stride = videoframe.get_stride_in_bytes();
                video_data = videoframe.get_data();
                
                depth_data = depthFrame.get_data();
                
                if (!video_data || !depth_data) 
                    std::cerr << "Invalid frames!" << std::endl;
                else{
                    color_frame = 1;
                    depth_frame_flag = 1;
                }
        
            }
            
            auto posNext = playback.get_position();
            if (posNext < posCurr) break;
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
        run_cam_thread++;
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

    // Create the second thread
    if (pthread_create(&ransac_thread, NULL, RANSAC_INLIERS, NULL) != 0) {
        std::cerr << "Error creating RANSAC_INLIERS thread!" << std::endl;
        return -1;
    }

    // Wait for threads to finish
    pthread_join(update_thread, NULL);
    pthread_join(frames_thread, NULL);
    pthread_join(points_thread, NULL);
    pthread_join(opengl_thread, NULL);
    pthread_join(ransac_thread, NULL);

    // run_camera();

    return 0;
}
