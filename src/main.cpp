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
#include <queue>
#include <tuple>
#include <iostream>
#include <cstdlib>   // setenv
#include <AL/al.h>
#include <AL/alc.h>
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>
#include "visualization/visualization.hpp"
#include "camera/camera.hpp"
#include "ransac/ransac.hpp"
#include "kdtree/kdtree.hpp"
#include <map>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> // Include GLM header for matrix operations
#include <glm/gtx/quaternion.hpp>

#include <Eigen/Dense>
#include <condition_variable>
#include <unordered_set>

#include <algorithm>    // std::find
#include <numeric>      // std::accumulate
#include <cmath>        // std::sqrt
#include <limits>       // std::numeric_limits
#include <cstdio>
#include <fcntl.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define MODEL_PATH   "/home/rick/Misc/TestingCode/ssd_mobilenet_v2.tflite"
#define LABEL_PATH   "/home/rick/Misc/TestingCode/coco_labels.txt"
#define IMAGE_PATH   "/home/rick/Misc/TestingCode/test.png"
#define CONF_THRESH  0.5

// Input/output details
int input_tensor; 
TfLiteIntArray* dims; 
int height_tensor; 
int width_tensor;
int channels_tensor;
std::unique_ptr<tflite::Interpreter> interpreter;
std::vector<std::string> labels;

int suppress_stderr() {
    int fd = dup(STDERR_FILENO);
    int nullfd = open("/dev/null", O_WRONLY);
    dup2(nullfd, STDERR_FILENO);
    close(nullfd);
    return fd;  // save original stderr to restore later
}

void restore_stderr(int fd) {
    dup2(fd, STDERR_FILENO);
    close(fd);
}

std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();


class AudioPlayer {
    public:
        AudioPlayer(int frequency, int duration)
            : frequency(frequency), duration(duration), source(0), buffer(0) {
    
            // Initialize OpenAL
            initOpenAL();
        }
    
        ~AudioPlayer() {
            cleanup();
        }
    
        void play() {
            // Generate the sine wave for the beep
            auto beepSound = generateSineWave(frequency, duration, SAMPLE_RATE);
    
            // Load data into buffer
            alBufferData(buffer, AL_FORMAT_MONO16, beepSound.data(), beepSound.size() * sizeof(ALshort), SAMPLE_RATE);
    
            // Attach buffer to source
            alSourcei(source, AL_BUFFER, buffer);
            alSourcef(source, AL_GAIN, 1.0f);  // Set the volume
    
            // Play the beep
            alSourcePlay(source);
            // std::cout << "Playing beep sound...\n";
        }
    
        void pause() {
            alSourcePause(source);
            // std::cout << "Paused beep sound\n";
        }
    
        void resume() {
            alSourcePlay(source);
            // std::cout << "Resumed beep sound\n";
        }
        
        void panRightToLeft(float startX = 5.0f, float endX = -5.0f, float duration = 3.0f) {
            const int steps = 100;
            float stepSize = (endX - startX) / steps;
            float sleepTime = duration / steps;
        
            // Set listener at origin
            ALfloat listenerPos[] = { 0.0f, 0.0f, 0.0f };
            ALfloat listenerOri[] = { 0.0f, 0.0f, -1.0f,  // "at" vector
                                      0.0f, 1.0f,  0.0f }; // "up" vector
            alListenerfv(AL_POSITION, listenerPos);
            alListenerfv(AL_ORIENTATION, listenerOri);
        
            // Set initial source position
            ALfloat sourcePos[] = { startX, 0.0f, 0.0f };
            alSourcefv(source, AL_POSITION, sourcePos);
        
            alSourcePlay(source);  // Play the sound
        
            for (int i = 0; i <= steps; ++i) {
                sourcePos[0] = startX + i * stepSize;
                alSourcefv(source, AL_POSITION, sourcePos);
                std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
            }
        
            // Wait for the sound to finish playing
            ALint state;
            do {
                alGetSourcei(source, AL_SOURCE_STATE, &state);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } while (state == AL_PLAYING);
        }
    
        void playFromPosition(float x, float y, float z) {
            // Generate the sine wave
            auto beepSound = generateSineWave(frequency, duration, SAMPLE_RATE);
            
            // Load data into buffer
            alBufferData(buffer, AL_FORMAT_MONO16, beepSound.data(), beepSound.size() * sizeof(ALshort), SAMPLE_RATE);
        
            // Attach buffer to source (ensure it's bound)
            alSourcei(source, AL_BUFFER, buffer);
        
            // Set the source position
            ALfloat sourcePos[] = { x, y, z };
            alSourcefv(source, AL_POSITION, sourcePos);
        
            // Reset and play
            alSourceStop(source);
            alSourceRewind(source);
            alSourcePlay(source);
        
            // std::cout << "Playing sound from position: (" << x << ", " << y << ", " << z << ")\n";
        }
        
        
        ALCcontext* getContext() const {
            return context;
        }
    
    private:
        void initOpenAL() {
            setenv("ALSOFT_DRIVERS", "alsa", 1);         // force ALSA
            setenv("ALSA_CARD", "Headphones", 1);        // use headphones card
    
            device = alcOpenDevice(nullptr);  // default device
            if (!device) {
                std::cerr << "Failed to open audio device!" << std::endl;
                exit(-1);
            }
    
            context = alcCreateContext(device, nullptr);
            if (!context || !alcMakeContextCurrent(context)) {
                std::cerr << "Failed to create or make context current!" << std::endl;
                exit(-1);
            }
    
            std::cout << "OpenAL initialized with default (Headphones)!\n";
    
            // Generate the buffer and source
            alGenBuffers(1, &buffer);
            alGenSources(1, &source);
        }
    
        void cleanup() {
            // Clean up resources
            alDeleteSources(1, &source);
            alDeleteBuffers(1, &buffer);
            alcMakeContextCurrent(nullptr);
            alcDestroyContext(context);
            alcCloseDevice(device);
        }
    
        std::vector<ALshort> generateSineWave(int frequency, int duration, int sampleRate) {
            int samples = duration * sampleRate;
            std::vector<ALshort> buffer(samples);
    
            for (int i = 0; i < samples; ++i) {
                float sample = std::sin(2.0 * M_PI * frequency * i / sampleRate);
                buffer[i] = static_cast<ALshort>(sample * 32767 * VOLUME); // Scale the sample
            }
            return buffer;
        }
    
        const int SAMPLE_RATE = 44100;  // Sample rate
        const float VOLUME = 0.1f;      // Volume of the beep (0.0 to 1.0)
        ALCdevice* device;
        ALCcontext* context;
        ALuint source;
        ALuint buffer;
        int frequency;
        int duration;
    };

// Custom hash function for Eigen::Vector3d
struct Vector3dHash {
    std::size_t operator()(const Eigen::Vector3d& v) const {
        std::hash<double> hasher;
        return hasher(v.x()) ^ (hasher(v.y()) << 1) ^ (hasher(v.z()) << 2);
    }
};

// Custom equality comparison for Eigen::Vector3d
struct Vector3dEqual {
    bool operator()(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) const {
        return lhs.isApprox(rhs, 1e-6); // Allows small floating-point differences
    }
};

// Approximate search using std::any_of
bool isCloseEnough(const point_t& a, const point_t& b, double epsilon = 1e-6) {
  for (size_t i = 0; i < a.size(); i++) {
      if (std::fabs(a[i] - b[i]) > epsilon) return false;
  }
  return true;
}

// Hash function for unordered_set
struct PointHash {
  size_t operator()(const point_t& p) const {
      std::hash<double> hasher;
      return hasher(p[0]) ^ hasher(p[1]) ^ hasher(p[2]);  // Simple hash combination
  }
};

struct PointEqual {
  bool operator()(const point_t& a, const point_t& b) const {
      return isCloseEnough(a, b);
  }
};

// Optimized unordered_set method
bool containsPointFast(const std::unordered_set<point_t, PointHash, PointEqual>& cloudSet, const point_t& query) {
    return cloudSet.find(query) != cloudSet.end();
}

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
const void* video_data;

const void* depth_data;
uint16_t *depth;

uint8_t *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;

// uint8_t *depth_front;
// uint8_t *rgb_front;

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
pthread_t remove_floor;
pthread_t audio_thread;
pthread_t spatial_audio;
pthread_t objectDetection;


pthread_t cleanUp_thread;
// Create the second thread


pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;


int window = 0;
GLuint gl_depth_tex;

int rgbWindow = 0;
GLuint gl_rgb_tex;

// Intrinsic parameters for intel 
const double fx = 428.299;  // Focal length x
const double fy = 428.299;  // Focal length y
const double cy = 244.483;   // Principal point x
const double cx = 419.197;   // Principal point y

rs2::pointcloud pc;  // Point cloud object
rs2::points points;  // Container for calculated point cloud data
rs2::frame_queue filtered_data;
rs2::frame_queue rgbframe_data;
rs2::frame_queue gyro_data_queue;
rs2::frame_queue acce_data_queue;

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

float ax, ay, az , gx, gy, gz = 0;

uint8_t got_depth = 0;
uint8_t got_Inliers = 0;

// Shared data structure
struct SharedData {
    std::vector<Eigen::Vector3d> pointCloud; // Point cloud data
    std::vector<Eigen::Vector3d> inliers;    // Inliers from RANSAC
    bool pointCloudReady = false;            // Flag to indicate new point cloud is available
    std::mutex mutex;                        // Mutex for thread-safe access
};

SharedData sharedData; // Global shared data

// Global variables for arcball rotation
glm::quat rotation = glm::quat(-.064675, 0.782132, 0.019156, -0.372425);  // identity quaternion
int windowWidth = 640;   // Update these with your window dimensions
int windowHeight = 480;

// Global buffers for point cloud double-buffering.
std::vector<PointCloudData> pointCloudBuffer[2];
int activeBufferIndex = 0; // Buffer used by the renderer.
std::atomic<bool> point_cloud_available(false);
std::atomic<bool> point_cloud_data(false);

// Global frame counter.
std::atomic<uint32_t> frame_number(0);

// Mutex to protect shared access.
std::mutex localPointCloudMutex;

bool updated_ = false;
std::queue<std::tuple<rs2::frame, rs2::frame, rs2::depth_frame, rs2::frame>> frameQueue;
std::mutex frameQueueMutex;

std::mutex mtx;
std::condition_variable cond_v;
bool ready = true;
int current_thread = 1; // Start with thread1

int frame = 1;
uint8_t got_rgb = 0;


class GyroBias
{
  private:
    int calibrationUpdates;
    double minX, maxX;
    double minY, maxY;
    double minZ, maxZ;


  public:
    bool isSet;

    double x;
    double y;
    double z;

    GyroBias()
    {
        reset();
    }

    void reset()
    {
        calibrationUpdates = 0;

        minX = 1000;
        minY = 1000;
        minZ = 1000;
        maxX = -1000;
        maxY = -1000;
        maxZ = -1000;

        x = 0;
        y = 0;
        z = 0;

        isSet = false;
    }


    bool update(double gx, double gy, double gz)
    {
        if (calibrationUpdates < 50)
        {   
            maxX = std::max(gx, maxX);
            maxY = std::max(gy, maxY);
            maxZ = std::max(gz, maxZ);

            minX = std::min(gx, minX);
            minY = std::min(gy, minY);
            minZ = std::min(gz, minZ);

            calibrationUpdates++;
            return false;
        }
        else if (calibrationUpdates == 50)
        {
            x = (maxX + minX)/2.0;
            y = (maxY + minY)/2.0;
            z = (maxZ + minZ)/2.0;
            calibrationUpdates++;

        
            isSet = true;

            return true;
        }
        else
        {
            return false;
        }
    }
};

int mouseX = 0, mouseY = 0;

float mouseWorldX = 0, mouseWorldY = 0 , mouseWorldZ = 0;

void drawBoundingBox(const BoundingBox& box) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set transparent color (e.g., blue with 30% opacity)
    glColor4f(0.0f, 0.0f, 1.0f, 0.3f);  

    glBegin(GL_QUADS);
    // Front face
    glVertex3f(box.x_min, box.y_min, box.z_max);
    glVertex3f(box.x_max, box.y_min, box.z_max);
    glVertex3f(box.x_max, box.y_max, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_max);

    // Back face
    glVertex3f(box.x_min, box.y_min, box.z_min);
    glVertex3f(box.x_max, box.y_min, box.z_min);
    glVertex3f(box.x_max, box.y_max, box.z_min);
    glVertex3f(box.x_min, box.y_max, box.z_min);

    // Left face
    glVertex3f(box.x_min, box.y_min, box.z_min);
    glVertex3f(box.x_min, box.y_min, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_min);

    // Right face
    glVertex3f(box.x_max, box.y_min, box.z_min);
    glVertex3f(box.x_max, box.y_min, box.z_max);
    glVertex3f(box.x_max, box.y_max, box.z_max);
    glVertex3f(box.x_max, box.y_max, box.z_min);

    // Top face
    glVertex3f(box.x_min, box.y_max, box.z_min);
    glVertex3f(box.x_max, box.y_max, box.z_min);
    glVertex3f(box.x_max, box.y_max, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_max);

    // Bottom face
    glVertex3f(box.x_min, box.y_min, box.z_min);
    glVertex3f(box.x_max, box.y_min, box.z_min);
    glVertex3f(box.x_max, box.y_min, box.z_max);
    glVertex3f(box.x_min, box.y_min, box.z_max);
    glEnd();

    // Draw wireframe edges for clarity
    glColor4f(0.0f, 0.0f, 1.0f, 1.0f); // Solid blue edges
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // Connect the corners to draw a cube outline
    glVertex3f(box.x_min, box.y_min, box.z_min); glVertex3f(box.x_max, box.y_min, box.z_min);
    glVertex3f(box.x_min, box.y_max, box.z_min); glVertex3f(box.x_max, box.y_max, box.z_min);
    glVertex3f(box.x_min, box.y_min, box.z_max); glVertex3f(box.x_max, box.y_min, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_max); glVertex3f(box.x_max, box.y_max, box.z_max);
    
    glVertex3f(box.x_min, box.y_min, box.z_min); glVertex3f(box.x_min, box.y_max, box.z_min);
    glVertex3f(box.x_max, box.y_min, box.z_min); glVertex3f(box.x_max, box.y_max, box.z_min);
    glVertex3f(box.x_min, box.y_min, box.z_max); glVertex3f(box.x_min, box.y_max, box.z_max);
    glVertex3f(box.x_max, box.y_min, box.z_max); glVertex3f(box.x_max, box.y_max, box.z_max);
    
    glVertex3f(box.x_min, box.y_min, box.z_min); glVertex3f(box.x_min, box.y_min, box.z_max);
    glVertex3f(box.x_max, box.y_min, box.z_min); glVertex3f(box.x_max, box.y_min, box.z_max);
    glVertex3f(box.x_min, box.y_max, box.z_min); glVertex3f(box.x_min, box.y_max, box.z_max);
    glVertex3f(box.x_max, box.y_max, box.z_min); glVertex3f(box.x_max, box.y_max, box.z_max);
    glEnd();
}


void GetMouseMotion(int x, int y) {
    mouseX = x;
    mouseY = windowHeight - y;  // Convert to OpenGL coordinate system

    // Read the depth buffer at the mouse position
    float depth;
    glReadPixels(mouseX, mouseY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

    // Convert depth to world coordinates
    glm::vec3 screenPos(mouseX, mouseY, depth);
    glm::mat4 projection, view;
    glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(projection));
    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(view));
    
    glm::vec4 viewport(0, 0, windowWidth, windowHeight);
    glm::vec3 worldPos = glm::unProject(screenPos, view, projection, viewport);

    // Store world coordinates for rendering
    mouseWorldX = worldPos.x;
    mouseWorldY = worldPos.y;
    mouseWorldZ = worldPos.z;
}
    
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


void renderText(float x, float y, const char *text, void *font) {
    glRasterPos2f(x, y); // Position for text rendering
    while (*text) {
        glutBitmapCharacter(font, *text);
        text++;
    }
}


void drawGrid(float size = 10.0f, float step = 1.0f) {
    glColor4f(0.3f, 0.3f, 0.3f, 0.35f); // Dark gray color for the grid
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

// Helper function to draw a small transparent cube centered at the origin
void drawTransparentCube() {
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Set the cube color with 50% opacity (red, for example)
    glColor4f(1.0f, 0.0f, 0.0f, 0.5f);
    
    // Define the cube with side length 0.5 (adjust as needed)
    float halfSide = 0.05f; // half the side length
    
    glBegin(GL_QUADS);
    // Front face (z = +halfSide)
    glVertex3f(-halfSide, -halfSide,  halfSide);
    glVertex3f( halfSide, -halfSide,  halfSide);
    glVertex3f( halfSide,  halfSide,  halfSide);
    glVertex3f(-halfSide,  halfSide,  halfSide);

    // Back face (z = -halfSide)
    glVertex3f(-halfSide, -halfSide, -halfSide);
    glVertex3f(-halfSide,  halfSide, -halfSide);
    glVertex3f( halfSide,  halfSide, -halfSide);
    glVertex3f( halfSide, -halfSide, -halfSide);

    // Left face (x = -halfSide)
    glVertex3f(-halfSide, -halfSide, -halfSide);
    glVertex3f(-halfSide, -halfSide,  halfSide);
    glVertex3f(-halfSide,  halfSide,  halfSide);
    glVertex3f(-halfSide,  halfSide, -halfSide);

    // Right face (x = +halfSide)
    glVertex3f( halfSide, -halfSide, -halfSide);
    glVertex3f( halfSide,  halfSide, -halfSide);
    glVertex3f( halfSide,  halfSide,  halfSide);
    glVertex3f( halfSide, -halfSide,  halfSide);

    // Top face (y = +halfSide)
    glVertex3f(-halfSide,  halfSide, -halfSide);
    glVertex3f(-halfSide,  halfSide,  halfSide);
    glVertex3f( halfSide,  halfSide,  halfSide);
    glVertex3f( halfSide,  halfSide, -halfSide);

    // Bottom face (y = -halfSide)
    glVertex3f(-halfSide, -halfSide, -halfSide);
    glVertex3f( halfSide, -halfSide, -halfSide);
    glVertex3f( halfSide, -halfSide,  halfSide);
    glVertex3f(-halfSide, -halfSide,  halfSide);
    glEnd();
    
    glDisable(GL_BLEND);
}

double computeMeanDistance(const pointVec& c1, const pointVec& c2){
    
    if (c1.empty() || c2.empty()) return 0.0;

    // For each point in c2, compute nearest distance to c1, then average.
    // (Or do a full NxM approach for clarity.)
    std::vector<double> distances;
    distances.reserve(c2.size() * c1.size());

    for (const auto& p2 : c2) {
    for (const auto& p1 : c1) {
    // Euclidean distance
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    distances.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    }
    // Average distance
    double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    return sum / static_cast<double>(distances.size());
}

// Define global/shared variables
std::vector<PointCloudData> filteredPointCloud;
uint8_t displayIn = 0;
uint8_t kd_algorithm = 0;

using Neiborhoods  =  std::vector<pointVec>;
pointVec points_kd;

float averageDistanceFromOrigin(const pointVec& cluster) {
    if (cluster.empty()) return 0.0f;

    float totalDistance = 0.0f;
    for (const auto& pt : cluster) {
        float d = std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
        totalDistance += d;
    }

    return totalDistance / static_cast<float>(cluster.size());
}

point_t computeCentroid(const pointVec& cluster) {
    point_t centroid = {0.0f, 0.0f, 0.0f};

    if (cluster.empty()) return centroid;

    for (const auto& pt : cluster) {
        centroid[0] += pt[0];
        centroid[1] += pt[1];
        centroid[2] += pt[2];
    }

    float size = static_cast<float>(cluster.size());
    centroid[0] /= size;
    centroid[1] /= size;
    centroid[2] /= size;

    return centroid;
}

#include <sstream>



uint8_t audio_flag = 1;
bool    audio_playing = false;
float seconds_ = 3;
float initial_seconds = 3;
std::string objText;
point_t center;

std::mutex rgb_mutex;


void InitRGBGL() {
    glutSetWindow(rgbWindow);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Allocate full‐size texture (1280×720)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
      std::cerr << "InitRGBGL: glTexImage2D error: " << err << std::endl;
    }
}

std::condition_variable  inference_cv;
std::mutex               inference_mutex;
bool                     newRgbAvailable = false;

// std::vector<std::string> labels;
std::mutex detection_mutex;
struct Detection { int xmin,ymin,xmax,ymax; int cls; float score; float distance; };
std::vector<Detection> detections;

void* inferenceWorker(void* arg) {
    cv::Mat small;
    cv::Mat full; // <-- Declare here so it's visible later
    while (!done) {
        // Wait for the right turn
        std::unique_lock<std::mutex> lock(mtx);
        cond_v.wait(lock, [] { return current_thread == 3 || done; });
        if (done) break;
        
        // Now, it's our turn (thread 3)
        lock.unlock();

        // --- inference logic ---

        {
            std::lock_guard<std::mutex> lock(rgb_mutex);
            if (!rgb_front) continue; // extra safety
            full = cv::Mat(480,640, CV_8UC3, rgb_front); // <-- Assign inside the mutex-protected block
            cv::resize(full, small, {width_tensor,height_tensor});
            cv::cvtColor(small, small, cv::COLOR_BGR2RGB);
            uint8_t* input_data = interpreter->typed_tensor<uint8_t>(input_tensor);
            memcpy(input_data, small.data, width_tensor * height_tensor * channels_tensor);
            // uint8_t* input_data = interpreter->typed_tensor<uint8_t>(input_tensor);
            // std::memcpy(interpreter->typed_tensor<uint8_t>(interpreter->inputs()[0]),
            //             small.data, width_tensor*height_tensor*channels_tensor);
            // cv::imshow("Detections", full);
            // cv::waitKey(0);
        }
        // Inference
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke TFLite model\n";
            // return -1;
        }
        
        // Output
        float* boxes = interpreter->typed_output_tensor<float>(0);
        float* classes = interpreter->typed_output_tensor<float>(1);
        float* scores  = interpreter->typed_output_tensor<float>(2);
        float* count   = interpreter->typed_output_tensor<float>(3);
        
        int orig_h = 480;
        int orig_w = 640;

        // Draw results
        int num = static_cast<int>(count[0]);
        int class_id = static_cast<int>(classes[0]);
        // std::cout<<num<<std::endl;
        {
            std::lock_guard<std::mutex> dl(detection_mutex);
            detections.clear();
            for (int i = 0; i < num; i++) {

                // std::cout << "Score: " << scores[i] << "\n";
                if (scores[i] >= CONF_THRESH){
                    
                    


                    int xmin = static_cast<int>(boxes[i * 4 + 1]*orig_w); 
                    int ymin = static_cast<int>(boxes[i * 4 + 0]*orig_h); // xmin, ymin
                    
                    int xmax = static_cast<int>(boxes[i * 4 + 3]*orig_w); 
                    int ymax = static_cast<int>(boxes[i * 4 + 2]*orig_h); // xmax, ymax
                    
                    float distance_ = 0;
                    double sum = 0.0;
                    int count = 0;

                    pointVec cluster;

                    for (int y = ymin; y < ymax; ++y) {
                        for (int x = xmin; x < xmax; ++x) {
                            int i = y * 640 + x;
                            int depth_value = (depth_front[3 * i + 0] << 8) | depth_front[3 * i + 1];
                            float z = depth_value * 0.001f;
                            if (z > 0.0f && z < 2.0f) {
                                float x_world = (x - cx) * z / fx;
                                float y_world = (y - cy) * z / fy;
                                cluster.push_back({x_world, y_world, z});
                            }
                        }
                    }
                    
                    distance_ = averageDistanceFromOrigin(cluster);


                    
                    Detection d = {
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        static_cast<int>(classes[i]), 
                        scores[i],
                        distance_
                    };
                    detections.push_back(d);
                }
        
            }
        }
    }
    std::cout << "Closing thread: inferenceWorker" << std::endl;
    pthread_exit(NULL);
    return NULL;
}


void DrawRGBWindow() {
    glutSetWindow(rgbWindow);
    glClear(GL_COLOR_BUFFER_BIT);

    std::lock_guard<std::mutex> lock(rgb_mutex);
    if (rgb_front) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
            640, 480,
        GL_RGB, GL_UNSIGNED_BYTE,
        rgb_front);

        // Setup ortho
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, 640, 0, 480, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Draw RGB image FIRST
        glColor3f(1,1,1);
        glBegin(GL_QUADS);
            glTexCoord2f(0,1); glVertex2f(  0,   0);
            glTexCoord2f(1,1); glVertex2f(640,   0);
            glTexCoord2f(1,0); glVertex2f(640, 480);
            glTexCoord2f(0,0); glVertex2f(  0, 480);
        glEnd();

        glDisable(GL_TEXTURE_2D);

        // THEN draw detection boxes on top
        std::lock_guard<std::mutex> lk(detection_mutex);
        for(auto &d : detections){
            int x1 = d.xmin;
            int y1 = d.ymin;
            int x2 = d.xmax;
            int y2 = d.ymax;

            glColor3f(1,0,0);
            glLineWidth(2);
            glBegin(GL_LINE_LOOP);
                glVertex2f(x1,y1); glVertex2f(x2,y1);
                glVertex2f(x2,y2); glVertex2f(x1,y2);
            glEnd();

            std::string txt = labels[d.cls] + " " + std::to_string(int(d.score*100)) + "%" + " " + std::to_string(d.distance) + " m";
            glColor3f(1,1,1);
            glRasterPos2f(x1, y2+5);
            for(char c: txt) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
        }

        // Restore
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    glutSwapBuffers();
}


void DrawGLScene() {

    // Ensure we’re rendering to the point‐cloud window:
    glutSetWindow(window);

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
    drawGrid(5.0f, 1.0f);  

    // (Optional) Draw a transparent cube at the origin.
    glPushMatrix();
        drawTransparentCube();
    glPopMatrix();

    // --- Render the filtered point cloud (excluding inliers) ---
    std::vector<GLfloat> mainVertices;
    std::vector<GLfloat> mainColors;
    {
        std::lock_guard<std::mutex> lock(localPointCloudMutex);
        for (const auto& pt : filteredPointCloud) {
            mainVertices.push_back(pt.position.x());
            mainVertices.push_back(pt.position.y());
            mainVertices.push_back(pt.position.z());
            
            mainColors.push_back(pt.color.x());
            mainColors.push_back(pt.color.y());
            mainColors.push_back(pt.color.z());
        }
    }
    if (!mainVertices.empty()) {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
    
        glVertexPointer(3, GL_FLOAT, 0, mainVertices.data());
        glColorPointer(3, GL_FLOAT, 0, mainColors.data());
    
        glDrawArrays(GL_POINTS, 0, mainVertices.size() / 3);
    
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
    
    // --- Render inliers in red ---
    // if (displayIn && !result.inliers.empty()) {
    //     std::vector<GLfloat> inlierVertices;
    //     std::vector<GLfloat> inlierColors;
    //     for (const auto& pt : result.inliers) {
    //         inlierVertices.push_back(static_cast<GLfloat>(pt.x()));
    //         inlierVertices.push_back(static_cast<GLfloat>(pt.y()));
    //         inlierVertices.push_back(static_cast<GLfloat>(pt.z()));

    //         // Red color for inliers
    //         inlierColors.push_back(1.0f);  // Red
    //         inlierColors.push_back(0.0f);  // Green
    //         inlierColors.push_back(0.0f);  // Blue
    //     }
    //     glEnableClientState(GL_VERTEX_ARRAY);
    //     glEnableClientState(GL_COLOR_ARRAY);
    
    //     glVertexPointer(3, GL_FLOAT, 0, inlierVertices.data());
    //     glColorPointer(3, GL_FLOAT, 0, inlierColors.data());
    
    //     glPointSize(5.0f);
    //     glDrawArrays(GL_POINTS, 0, inlierVertices.size() / 3);
    
    //     glDisableClientState(GL_VERTEX_ARRAY);
    //     glDisableClientState(GL_COLOR_ARRAY);
    // }

    std::lock_guard<std::mutex> lock(localPointCloudMutex);
    if(filteredPointCloud.size()> 0 && kd_algorithm){
        KDTree tree(points_kd);
        Neiborhoods clusters;
        pointVec Cluster;
        uint8_t firstCluster = 1;
        pointVec last_cloud;
        int min = 50;
        int max = 250; 
        float t = 0.55;

        for(auto const& point : filteredPointCloud){
            if(point.position[2] <= 1){
                
                point_t pt = {point.position[0],point.position[1],point.position[2]};

                pointVec neighborhood = tree.neighborhood_points(pt, t);
                
                float filter = averageDistanceFromOrigin(neighborhood);

                if(filter <= 1){
                    if(neighborhood.size() > min && neighborhood.size() < max){
                        if(clusters.size() == 0){
                            clusters.push_back(neighborhood);
                        }
                        else{
                            
                            for(auto const& clus : clusters){
                                double meanDist = computeMeanDistance(clus, neighborhood);
                                if (meanDist > 0.45) {
                                    clusters.push_back(neighborhood);
                                }
                            }
                            // std::cout<<meanDist<<std::endl;
                            // last_cloud = neighborhood;
                            
                            // firstCluster = 0;
                        }
                    }
                }
            }

            // if (clusters.size() > 0) break;
        }
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            // Simple color generation: cycle through a few preset colors.
            float r = 0.0f, g = 0.0f, b = 0.0f;
            switch (i % 6) {
                case 0: r = 1.0f; g = 1.0f; b = 0.0f; break; // red
                case 1: r = 0.0f; g = 1.0f; b = 0.0f; break; // green
                case 2: r = 0.0f; g = 0.0f; b = 1.0f; break; // blue
                case 3: r = 1.0f; g = 1.0f; b = 0.0f; break; // yellow
                case 4: r = 1.0f; g = 0.0f; b = 1.0f; break; // magenta
                case 5: r = 0.0f; g = 1.0f; b = 1.0f; break; // cyan
            }
            std::vector<GLfloat> clusterVertices;
            for (const auto& pt : clusters[i]) {
                // Since point_t is std::vector<double> with at least 3 elements:
                clusterVertices.push_back(static_cast<GLfloat>(pt[0]));
                clusterVertices.push_back(static_cast<GLfloat>(pt[1]));
                clusterVertices.push_back(static_cast<GLfloat>(pt[2]));
            }
            if (!clusterVertices.empty()) {
                glEnableClientState(GL_VERTEX_ARRAY);
                glVertexPointer(3, GL_FLOAT, 0, clusterVertices.data());
                glColor3f(r, g, b); // Set the current color for this cluster.
                glPointSize(7.0f); // Slightly larger for clusters.
                glDrawArrays(GL_POINTS, 0, clusterVertices.size() / 3);
                glDisableClientState(GL_VERTEX_ARRAY);
            }
            BoundingBox bbx = bounding_box(clusters[i]);
            drawBoundingBox(bbx);

            // --- Show object distance ---
            float avgDist = averageDistanceFromOrigin(clusters[i]);
            
            avgDist = std::round(avgDist*10.0)/10.0;
            // Round and format avgDist
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << avgDist;
            
            objText = "Object detected: " + std::to_string(avgDist).substr(0, 3) + " meters from camera";
            
            center = computeCentroid(clusters[i]);
            
            if(!audio_playing){
                audio_flag = 1;
            }
            
            // Draw text on screen
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();

            glColor3f(1.0f, 0.0f, 0.0f); // Red text
            renderText(20, windowHeight - 60 - (int)i * 20, objText.c_str(), GLUT_BITMAP_HELVETICA_18);

            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);

            glColor3f(r, g, b); // Use the same color as the cluster
            glLineWidth(2.0f);
            glBegin(GL_LINES);
                glVertex3f(0.0f, 0.0f, 0.0f);           // Origin
                glVertex3f(center[0], center[1], center[2]); // Cluster centroid
            glEnd();

        }
        clusters.clear();
    }

    // -------------------- Draw UI Elements --------------------
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    glColor3f(0.0f, 0.0f, 0.0f);
    std::string quatText = "Rotation (quat): w: " + std::to_string(rotation.w) +
                             " x: " + std::to_string(rotation.x) +
                             " y: " + std::to_string(rotation.y) +
                             " z: " + std::to_string(rotation.z);
    renderText(20, windowHeight - 20, quatText.c_str(), GLUT_BITMAP_HELVETICA_18);
    point_cloud_available = false;
    
    std::string mouseText = "Mouse 3D: (" + std::to_string(mouseWorldX) + ", " 
                                     + std::to_string(mouseWorldY) + ", " 
                                     + std::to_string(mouseWorldZ) + ")";
    renderText(20, windowHeight - 40, mouseText.c_str(), GLUT_BITMAP_HELVETICA_18);


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
        exit(0);
        glutLeaveMainLoop();
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
    glutPassiveMotionFunc(GetMouseMotion);
    glutKeyboardFunc(keyboard);
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0);

    // — RGB window —
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(1300, 100);
    rgbWindow = glutCreateWindow("RGB Frame Viewer");
    glutDisplayFunc(&DrawRGBWindow);
    InitRGBGL();  // your RGB GL init (rgb texture, etc.)

    // — Single idle to drive BOTH windows —
    glutIdleFunc([](){
        // queue redraw of point‑cloud
        glutSetWindow(window);
        glutPostRedisplay();
        // queue redraw of RGB
        glutSetWindow(rgbWindow);
        glutPostRedisplay();
    });

    // Enter the GLUT main loop (never returns)
    glutMainLoop();
    return NULL;
}


// void InitRGBGL() {
//     // Make sure we’re on the RGB window’s context:
//     glutSetWindow(rgbWindow);

//     glEnable(GL_TEXTURE_2D);
//     glGenTextures(1, &gl_rgb_tex);
//     glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//     // Allocate an empty texture of the correct size so future uploads work:
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 848, 480, 0,
//                  GL_RGB, GL_UNSIGNED_BYTE, nullptr);
//     GLenum err = glGetError();
//     if (err != GL_NO_ERROR) {
//       std::cerr << "InitRGBGL: glTexImage2D error: " << err << std::endl;
//     }
// }

// void InitGL()
// {
//     glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set to white background
//     glEnable(GL_TEXTURE_2D);
//     glGenTextures(1, &gl_depth_tex);
//     glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     // ReSizeGLScene(640, 480);
// }

std::mutex audioMutex;  // Mutex for thread safety of shared variables

bool  played = false;
bool  canplay_ = false;
float   timer_ = 0.5;
void* SpatialAudio(void* arg){
    // Audio manager
    
    AudioPlayer player(440, 1);  // 440Hz for 3 seconds
    int old_stderr = suppress_stderr();
    restore_stderr(old_stderr);
    alcMakeContextCurrent(player.getContext()); 
    while(!done) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta = now - lastTime;
        float deltaTime = delta.count();
        lastTime = now;

        std::lock_guard<std::mutex> lock(audioMutex);
        
        if(audio_flag && !played){
            played = true;
        }

        if(played){
            player.playFromPosition(center[0], center[1], center[2]);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    
    }
    std::cout << "Closing thread: SpatialAudio" << std::endl;
    pthread_exit(NULL);
    return NULL;
}

void* AudioFeedback(void* arg) {
    int old_stderr = suppress_stderr();
    while(!done) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta = now - lastTime;
        float deltaTime = delta.count();
        lastTime = now;

        // Check if audio should play
        {
            std::lock_guard<std::mutex> lock(audioMutex);  // Protect shared resources

            if(audio_flag && !audio_playing) {  // Ensure no overlap in audio playback
                audio_playing = true;  // Mark that audio is playing
                audio_flag = 0;  // Reset the flag to avoid playing multiple times
                // objText
                std::string command = "espeak \"" + objText + "\"";
                // Execute the command
                int result = system((command + " 2>/dev/null").c_str());

                // int result = system(command.c_str());
                restore_stderr(old_stderr);
                
            }
        }

        if(audio_playing) {
            seconds_ -= deltaTime;
        }

        // Add a cooldown period after each audio playback to avoid repeated triggering
        if(seconds_ <= 0) {
            std::lock_guard<std::mutex> lock(audioMutex);
            seconds_ = initial_seconds;  // Reset the timer
            audio_playing = false;  // Stop "playing" the audio
        }
    }
    std::cout << "Closing thread: AudioFeedback" << std::endl;
    pthread_exit(NULL);
    return NULL;
}




void* RemoveInliers(void* arg) {
    while (!done) {
        {
            // Lock access to the shared point cloud buffer.
            std::lock_guard<std::mutex> lock(localPointCloudMutex);

            if (got_Inliers) { // Ensure RANSAC has produced inliers.
                filteredPointCloud.clear();

                // Convert inliers list to a set for fast lookup.
                std::unordered_set<Eigen::Vector3d, Vector3dHash> inlierSet(
                    result.inliers.begin(), result.inliers.end());
                    
                    // Iterate through the original point cloud.
                    points_kd.clear();
                    for (const auto& point : pointCloudBuffer[activeBufferIndex]) {
                        // Convert PointCloudData to an Eigen::Vector3d for comparison.
                        Eigen::Vector3d pointPos(point.position[0],
                                                point.position[1],
                                                point.position[2]);
                                             
                        if (inlierSet.find(pointPos) == inlierSet.end()) {
                            // If the point is NOT an inlier, add the full PointCloudData,
                            // preserving both position and color.
                            filteredPointCloud.push_back(point);
                            points_kd.push_back({point.position[0], point.position[1], point.position[2]});
                        }
                    }
                    
                    got_Inliers = 0; // Reset flag after processing.
                    kd_algorithm = 1;
                    // // Hazard Detection
            }

        }

        // Small sleep to prevent busy-waiting.
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Closing thread: RemoveInliers" << std::endl;
    pthread_exit(NULL);
    return NULL;
}


void* RANSAC_INLIERS(void* arg) {
    while (!done) {
    {
        
        // Lock to safely access the shared point cloud buffer
        std::lock_guard<std::mutex> lock(localPointCloudMutex);
        if (point_cloud_available) {
            // Use the active point cloud from the current buffer.
            // Assuming pointCloudBuffer is a std::vector of your custom point type,
            // e.g., std::vector<PointXYZRGB>, that fits the signature of fit_plane_ransac.
            auto& currentPointCloud = pointCloudBuffer[activeBufferIndex];
            
            // Check if the point cloud is not empty before running RANSAC
            if (!currentPointCloud.empty()) {
                // Run RANSAC on the current point cloud.
                // For example, fit_plane_ransac may have a signature like:
                // RANSACResult fit_plane_ransac(const std::vector<PointXYZRGB>& cloud, int iterations, int threshold, Eigen::Vector3d axis);
                result = fit_plane_ransac(currentPointCloud, 0.15, 220, Eigen::Vector3d(0, 1, 0));
                
                // std::cout << "\rRANSAC inliers: " << result.inliers.size() << std::flush;
            }
            // Reset the flag to indicate we've processed the current point cloud
            ransac_data = 0;
            got_Inliers = 1;
            displayIn = 1;
        }
    }
    // Small sleep to prevent busy-waiting; adjust as needed.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "closing thread: RANSAC_INLIERS" << std::endl;
    pthread_exit(NULL);
    return NULL;
}



bool shutdown_ = false;
float roll = 0.0;
float pitch = 0.0;
float yaw = 0.0;

GyroBias bias;

bool firstAccel = true;
double last_ts[RS2_STREAM_COUNT];
double dt_[RS2_STREAM_COUNT];
double ts_;

void* pointcloud_generate(void* arg) {

    float d_width = 640.0f;
    float d_height = 480.0f;

    float rgb_width = 640.0f;
    float rgb_height  = 480.0f;


    const float scaleX = rgb_width / d_width;
    const float scaleY = rgb_height/ d_height;
    const int step_x = 15;
    const int step_y = 15;



    while (true) {
        if(shutdown_) break;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cond_v.wait(lock, [] { return current_thread == 3; });
        }
        
        // Simulate more work in main thread
        std::lock_guard<std::mutex> lock(localPointCloudMutex);
        int writeBuffer = 1 - activeBufferIndex;
        pointCloudBuffer[writeBuffer].clear();
        
        // Transform the cloud according to built in IMU (to get it straight)
        Eigen::Affine3f rx = Eigen::Affine3f(Eigen::AngleAxisf(-(pitch - M_PI_2), Eigen::Vector3f(1, 0, 0)));
        Eigen::Affine3f ry = Eigen::Affine3f(Eigen::AngleAxisf(0.0, Eigen::Vector3f(0, 1, 0)));
        Eigen::Affine3f rz = Eigen::Affine3f(Eigen::AngleAxisf(roll - M_PI_2, Eigen::Vector3f(0, 0, 1)));
        Eigen::Affine3f rot = rz * ry * rx;

        for (int y = 0; y < d_height; y += step_y) {
            for (int x = 0; x < d_width; x += step_x) {
                int i = y * d_width + x;
                int depth_value = (depth_front[3 * i + 0] << 8) | depth_front[3 * i + 1];

                if (depth_value > 0 && depth_value < 2500) {
                    
                    float z = depth_value * 0.001f;
                    float x_world = (x - cx) * z / fx;
                    float y_world = (y - cy) * z / fy;

                
                    Eigen::Vector3f raw_points( x_world, y_world, z );
                    Eigen::Vector3f pt_transformed = rot * raw_points;


                    int rgb_x = static_cast<int>(x * scaleX);
                    int rgb_y = static_cast<int>(y * scaleY);
                    int rgb_idx = (rgb_y * rgb_width + rgb_x) * 3;

                    Eigen::Vector3f color(
                        rgb_front[rgb_idx + 0] / 255.0f,
                        rgb_front[rgb_idx + 1] / 255.0f,
                        rgb_front[rgb_idx + 2] / 255.0f
                    );

                    pointCloudBuffer[writeBuffer].emplace_back(
                        pt_transformed.x(), pt_transformed.y(), pt_transformed.z(),
                        color.x(), color.y(), color.z()
                    );
                }
            }
        }
        
        
        activeBufferIndex = writeBuffer;
        point_cloud_available = true;
        point_cloud_data = false;

        frame_number++;
        if (frame_number % 10 == 0) {
            std::cout << "\rFrame number: " << frame_number
                        << " | Point cloud size: " 
                        << pointCloudBuffer[writeBuffer].size()
                        << "     " << std::flush;
        }

        // std::cout<<"Frame " << frame << " Point Cloud Generated:  " <<"\n"; 
        
        frame++;
        {
            std::lock_guard<std::mutex> lock(mtx);
            current_thread = 1; // Hand over execution
            ready = true;
        }

        cond_v.notify_all();    
        
    }
    
    std::cout << "closing thread: pointcloud_generate" << std::endl;
    pthread_exit(NULL);
    return NULL;
}



// void* display_rgb_thread(void* arg) {
//     cv::namedWindow("RGB Frame", cv::WINDOW_NORMAL);
//     cv::resizeWindow("RGB Frame", 848, 480);

//     while (true) {
//         if (shutdown_) break;

//         {
//             std::unique_lock<std::mutex> lock(mtx);
//             cond_v.wait(lock, [] { return current_thread == 2; });  // Wait for RGB frame to be ready
//         }

//         // Lock RGB buffer
//         {
//             std::lock_guard<std::mutex> rgb_lock(rgb_mutex);  // <- mutex for rgb_front
//             if (!rgb_front) {
//                 std::cerr << "RGB front is null!" << std::endl;
//                 continue;
//             }

//             cv::Mat rgb(720, 1280, CV_8UC3, rgb_front);
//             cv::Mat rgb_resized;

//             // Resize to 848x480 for display
//             cv::resize(rgb, rgb_resized, cv::Size(848, 480));
//             // cv::cvtColor(rgb_resized, bgr_resized, cv::COLOR_RGB2BGR);

//             // cv::imshow("RGB Frame", rgb_resized);
//             // cv::waitKey(1);
//         }

//         // Do not advance thread ownership; just sleep until next trigger
//     }

//     std::cout << "closing thread: display_rgb_image" << std::endl;
//     pthread_exit(NULL);
//     return NULL;
// }


void* process_frame_raw_data(void* arg) {

    while(true){
        
        if(shutdown_) break;

        {
            std::unique_lock<std::mutex> lock(mtx);
            cond_v.wait(lock, [] { return current_thread == 2; });
        }

        
        uint16_t *depth = (uint16_t *)depth_data;
        
        if(rgb_back != (uint8_t*)video_data){
            rgb_back = (uint8_t*)video_data;
            rgb_mid = rgb_back;
            got_rgb = 1;
        }

        if(got_rgb){
            std::lock_guard<std::mutex> lock(rgb_mutex);
            rgb_front = rgb_mid;
            got_rgb = 0;
        }

        for (int i = 0; i < 640 * 480; i++) {
            int pval = depth[i];
            depth_mid[3 * i + 0] = (pval >> 8) & 0xff;  // High byte (Red)
            depth_mid[3 * i + 1] = pval & 0xff;         // Low byte (Green)
            depth_mid[3 * i + 2] = 0;                   // Unused for now

        }

        
        gx = gyro_data.x;
        gy = gyro_data.y;
        gz = gyro_data.z;
        
        double ratePitch = gx - bias.x;
        double rateYaw = gy - bias.y;
        double rateRoll = gz - bias.z;

        // Transform to rad from rad/s
        ratePitch *= dt_[RS2_STREAM_GYRO];
        rateYaw *= dt_[RS2_STREAM_GYRO];
        rateRoll *= dt_[RS2_STREAM_GYRO];

        // ROLL - Around "blue" (forward), poisitive => right
        roll += rateRoll;

        // PITCH - Around "red" (right), positve => right
        pitch -= ratePitch; 

        // YAW - Around "green" (down), positive => right
        yaw += rateYaw;


        ax = acce_data.x;
        ay = acce_data.y;
        az = acce_data.z;

        float R = sqrtf(ax * ax + ay * ay + az * az);
        float newRoll = acos(ax / R);
        float newYaw = acos(ay / R);
        float newPitch = acos(az / R);

        if (firstAccel)
        {
            firstAccel = false;
            roll = newRoll;
            yaw = newYaw;
            pitch = newPitch;
        }
        else
        {
            // Compensate GYRO-drift with ACCEL
            roll = roll * 0.98 + newRoll * 0.02;
            yaw = yaw * 0.98 + newYaw * 0.02;
            pitch = pitch * 0.98 + newPitch * 0.02;
        }

        
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

        
        // std::cout<<"Frame " << frame << " buffers loaded\n" << std::flush;
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            current_thread = 3; // Hand over execution
            newRgbAvailable = true;
            
        }

        cond_v.notify_all(); // Notify the other thread

    }
    proc_thread++;
    std::cout<< "closing thread: process_frame_raw_data "<<std::endl;
    pthread_exit(NULL);
    return NULL;
}



void* process_filtered(void *arg){
    
    while(true){
        
        if(done && frameQueue.empty()){
            shutdown_ = true;
            break;
        }
        
        {
            std::unique_lock<std::mutex> lock(mtx);
            cond_v.wait(lock, [] { return current_thread == 1; }); // Wait for the signal
        }
        
        if (frameQueue.empty()) {
            continue;
        }

        
        
        auto frameSet = frameQueue.front();
        frameQueue.pop();

        // Extract frames
        auto gyro_frame = std::get<0>(frameSet);
        auto gyro_motion = gyro_frame.as<rs2::motion_frame>();
        auto acce_frame = std::get<1>(frameSet);
        auto acce_motion = acce_frame.as<rs2::motion_frame>();
        auto depthFrame = std::get<2>(frameSet);
        auto color = std::get<3>(frameSet);

        rs2::stream_profile profile = depthFrame.get_profile();

        ts_  = depthFrame.get_timestamp();
        dt_[profile.stream_type()] = (ts_ - last_ts[profile.stream_type()] ) / 1000.0;
        last_ts[profile.stream_type()] = ts_;
        

        // std::cout<< " ### "<<std::endl;
        // std::cout<< " GYRO  : "<< formatTimeStamp(gyro_motion.get_timestamp()) <<std::endl;
        // std::cout<< " ACCE  : "<< formatTimeStamp(acce_motion.get_timestamp()) <<std::endl;
        // std::cout<< " DEPTH : "<< formatTimeStamp(depthFrame.get_timestamp()) <<std::endl;
        // std::cout<< " COLOR : "<< formatTimeStamp(color.get_timestamp()) <<std::endl;
        // std::cout<< " ### " <<std::endl;

        // Depth data
        depth_data = depthFrame.get_data();
        
        // Color data
        auto videFrame = color.as<rs2::video_frame>(); 
        width = videFrame.get_width();
        height = videFrame.get_height();   
        stride = videFrame.get_stride_in_bytes();
        video_data = videFrame.get_data();

        // motion data
        gyro_data = gyro_motion.get_motion_data();
        acce_data = acce_motion.get_motion_data();
        
        // std::this_thread::sleep_for(std::chrono::seconds(0.1));
        // std::cout<<"\rFrame " << frame << " Input loaded\n";
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            current_thread = 2; // Hand over execution
            ready = false;
        }

        cond_v.notify_all(); // Notify the other thread
        
    }  

    std::cout << "closing thread: process_filtered\n" << std::endl;
    pthread_exit(NULL);
    return NULL;
} 


void* run_camera(void *arg){
    try {
        
        std::string bag_file = "/home/rick/Downloads/20250425_152236.bag"; //20250425_152236.bag -- 20250219_141918.bag
        auto pipe_video = std::make_shared<rs2::pipeline>();
        
        rs2::config cfg_video;
        
        //enable different modes
        cfg_video.enable_device_from_file(bag_file);      // from a file
        cfg_video.enable_stream(RS2_STREAM_DEPTH);
        cfg_video.enable_stream(RS2_STREAM_COLOR);
        cfg_video.enable_stream(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
        cfg_video.enable_stream(RS2_STREAM_ACCEL,RS2_FORMAT_MOTION_XYZ32F);
        // cfg_sensor.enable_stream(RS2_STREAM_POSE,RS2_FORMAT_MOTION_XYZ32F);
        

        // real time
        // rs2::pipeline p;
        
        
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
        // p.start();
        
        auto device = pipe_video->get_active_profile().get_device();
        
        if(device.as<rs2::playback>()){
            
            rs2::playback playback = device.as<rs2::playback>();
            playback.set_real_time(false);

            uint64_t posCurr = playback.get_position();
            
            config_flag = 1;
            // pipe->try_wait_for_frames(&frameset, 1000)
            while (pipe_video->try_wait_for_frames(&frameset, 1000) && !done) {
            // while(true){    
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cond_v.wait(lock, [] { return ready; }); // Wait for the signal
                }
                // std::cout<<"Updating frame\n";

                // #### Alternate ways to get a frameset ####
                // auto frameset = pipe_video->wait_for_frames(1000);
                // auto frameset_sensor = pipe_sensor->wait_for_frames(1000);
                
                // rs2::frameset frameset = p.wait_for_frames();

                try{
                    gyro_frame = frameset.first(RS2_STREAM_GYRO,RS2_FORMAT_MOTION_XYZ32F);
                    // auto gyro_motion = gyro_frame.as<rs2::motion_frame>();
                    // auto acce_motion = acce_frame.as<rs2::motion_frame>();

                    acce_frame = frameset.first(RS2_STREAM_ACCEL ,RS2_FORMAT_MOTION_XYZ32F);
                    color = frameset.get_color_frame();
                    
                    depthFrame = (frameset.get_depth_frame()).as<rs2::depth_frame>();
                    rs2::frame filtered_dec = dec_filter.process(depthFrame);
                    rs2::frame filtered_spa = spat_filter.process(filtered_dec);
                    rs2::frame filtered_tem = temp_filter.process(depthFrame);
                    
                    // std::lock_guard<std::mutex> lock(frameQueueMutex);
                    frameQueue.push(std::make_tuple(gyro_frame, acce_frame, filtered_tem, color));
                    
                }
                catch (const rs2::error& e) {
                    std::cerr << "RealSense error: " << e.what() << std::endl;
                }
                
                
                auto posNext = playback.get_position();
                if (posNext < posCurr){
                    std::cout<< "\nDone with playback\n"<<std::endl;
                    done = true;
                }

                if(done) break;
                posCurr = posNext;
                
            }
        }
        
        // while(true){
            
        // }

        // pipe_video->stop();
        std::cout<< "closing thread: run_camera"<<std::endl;
        pthread_exit(NULL);
    }


    catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << " RealSense Error: " << e.what() << std::endl;
    }

    return NULL;
}


int main(int argc, char* argv[]) {

    // Load labels
    // labels.push_back("background");
    std::ifstream labelFile(LABEL_PATH);
    std::string line;
    while (std::getline(labelFile, line)) {
        labels.push_back(line);
    }

    // Load model
    auto model = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return -1;
    }

    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return -1;
    }

    // Allocate tensors
    interpreter->AllocateTensors();

    // Input/output details
    input_tensor = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input_tensor)->dims;
    height_tensor = dims->data[1];
    width_tensor = dims->data[2];
    channels_tensor = dims->data[3];
    
    // CreateAndInitWindow();
    std::cout << "Intel RealSense SDK Version: " << RS2_API_VERSION_STR << std::endl;
    
    int old_stderr = suppress_stderr();

    restore_stderr(old_stderr);

 

    depth_mid = (uint8_t *)malloc(640 * 480 * 3);
    depth_front = (uint8_t *)malloc(640 * 480 * 3);
    
    rgb_back = (uint8_t*)malloc(640 * 480 * 3);
    rgb_mid = (uint8_t*)malloc(640 * 480 * 3);
	rgb_front = (uint8_t*)malloc(640 * 480 * 3);

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
    
    // // Create the second thread
    if (pthread_create(&frames_thread, NULL, process_frame_raw_data, NULL) != 0) {
        std::cerr << "Error creating process_frame_raw_data thread!" << std::endl;
        return -1;
    }

    // // Create the second thread
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

    // Create the second thread
    if (pthread_create(&remove_floor, NULL, RemoveInliers, NULL) != 0) {
        std::cerr << "Error creating REMOVE_INLIERS thread!" << std::endl;
        return -1;
    }

    // Create the second thread
    if (pthread_create(&audio_thread, NULL, AudioFeedback, NULL) != 0) {
        std::cerr << "Error creating REMOVE_INLIERS thread!" << std::endl;
        return -1;
    }

    // Create the second thread
    if (pthread_create(&spatial_audio, NULL, SpatialAudio, NULL) != 0) {
        std::cerr << "Error creating REMOVE_INLIERS thread!" << std::endl;
        return -1;
    }

    
    // Create the second thread
    if (pthread_create(&objectDetection, NULL, inferenceWorker, NULL) != 0) {
        std::cerr << "Error creating REMOVE_INLIERS thread!" << std::endl;
        return -1;
    }

    // Wait for threads to finish
    pthread_join(update_thread, NULL);
    pthread_join(frames_thread, NULL);
    pthread_join(points_thread, NULL);
    pthread_join(opengl_thread, NULL);
    pthread_join(ransac_thread, NULL);
    pthread_join(remove_floor, NULL);
    pthread_join(audio_thread, NULL);
    pthread_join(spatial_audio, NULL);
    // pthread_join(objectDetection, NULL);

    // run_camera();

    return 0;
}
