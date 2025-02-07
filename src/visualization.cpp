#include "visualization/visualization.hpp"
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glut.h>
#include <math.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include "libfreenect.h"
#include <opencv2/opencv.hpp>
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
#include <fstream>

bool kbhit() {
    termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return true;
    }

    return false;
}

// Function to parse RAM usage from /proc/meminfo
double get_ram_usage() {
    std::string line, key;
    long value, total_memory = 0, free_memory = 0;
    std::ifstream file("/proc/meminfo");
    if (file.is_open()) {
        while (getline(file, line)) {
            std::stringstream ss(line);
            ss >> key >> value;

            if (key == "MemTotal:") {
                total_memory = value;
            } else if (key == "MemAvailable:") {
                free_memory = value;
                break;
            }
        }
    }
    if (total_memory == 0) return 0.0;
    return 100.0 * (1.0 - double(free_memory) / double(total_memory));
}

void CreateAndInitWindow() {
    std::cout<<"Create Window"<<std::endl;
}
