#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h> 
#include <opencv2/opencv.hpp>
#include "camera/camera.hpp"
#include <iostream>


uint32_t get_user_selection(const std::string& prompt_message)
{
    std::cout << "\n" << prompt_message;
    uint32_t input;
    std::cin >> input;
    std::cout << std::endl;
    return input;
}

std::string get_sensor_name(const rs2::sensor& sensor)
{
    // Sensors support additional information, such as a human readable name
    if (sensor.supports(RS2_CAMERA_INFO_NAME))
        return sensor.get_info(RS2_CAMERA_INFO_NAME);
    else
        return "Unknown Sensor";
}

rs2::sensor get_a_sensor_from_a_device(const rs2::device& dev)
{
    // A rs2::device is a container of rs2::sensors that have some correlation between them.
    // For example:
    //    * A device where all sensors are on a single board
    //    * A Robot with mounted sensors that share calibration information

    // Given a device, we can query its sensors using:
    std::vector<rs2::sensor> sensors = dev.query_sensors();

    std::cout << "Device consists of " << sensors.size() << " sensors:\n" << std::endl;
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
    {
        std::cout << "  " << index++ << " : " << get_sensor_name(sensor) << std::endl;
    }

    uint32_t selected_sensor_index = get_user_selection("Select a sensor by index: ");

    // The second way is using the subscript ("[]") operator:
    if (selected_sensor_index >= sensors.size())
    {
        throw std::out_of_range("Selected sensor index is out of range");
    }

    return  sensors[selected_sensor_index];
}

rs2::stream_profile choose_a_streaming_profile(const rs2::sensor& sensor)
{
    // A Sensor is an object that is capable of streaming one or more types of data.
    // For example:
    //    * A stereo sensor with Left and Right Infrared streams that
    //        creates a stream of depth images
    //    * A motion sensor with an Accelerometer and Gyroscope that
    //        provides a stream of motion information

    // Using the sensor we can get all of its streaming profiles
    std::vector<rs2::stream_profile> stream_profiles = sensor.get_stream_profiles();

    // Usually a sensor provides one or more streams which are identifiable by their stream_type and stream_index
    // Each of these streams can have several profiles (e.g FHD/HHD/VGA/QVGA resolution, or 90/60/30 fps, etc..)
    //The following code shows how to go over a sensor's stream profiles, and group the profiles by streams.
    std::map<std::pair<rs2_stream, int>, int> unique_streams;
    for (auto&& sp : stream_profiles)
    {
        unique_streams[std::make_pair(sp.stream_type(), sp.stream_index())]++;
    }
    std::cout << "Sensor consists of " << unique_streams.size() << " streams: " << std::endl;
    for (size_t i = 0; i < unique_streams.size(); i++)
    {
        auto it = unique_streams.begin();
        std::advance(it, i);
        std::cout << "  - " << it->first.first << " #" << it->first.second << std::endl;
    }

    //Next, we go over all the stream profiles and print the details of each one
    std::cout << "Sensor provides the following stream profiles:" << std::endl;
    int profile_num = 0;
    for (rs2::stream_profile stream_profile : stream_profiles)
    {
        // A Stream is an abstraction for a sequence of data items of a
        //  single data type, which are ordered according to their time
        //  of creation or arrival.
        // The stream's data types are represented using the rs2_stream
        //  enumeration
        rs2_stream stream_data_type = stream_profile.stream_type();

        // The rs2_stream provides only types of data which are
        //  supported by the RealSense SDK
        // For example:
        //    * rs2_stream::RS2_STREAM_DEPTH describes a stream of depth images
        //    * rs2_stream::RS2_STREAM_COLOR describes a stream of color images
        //    * rs2_stream::RS2_STREAM_INFRARED describes a stream of infrared images

        // As mentioned, a sensor can have multiple streams.
        // In order to distinguish between streams with the same
        //  stream type we can use the following methods:

        // 1) Each stream type can have multiple occurances.
        //    All streams, of the same type, provided from a single
        //     device have distinct indices:
        int stream_index = stream_profile.stream_index();

        // 2) Each stream has a user-friendly name.
        //    The stream's name is not promised to be unique,
        //     rather a human readable description of the stream
        std::string stream_name = stream_profile.stream_name();

        // 3) Each stream in the system, which derives from the same
        //     rs2::context, has a unique identifier
        //    This identifier is unique across all streams, regardless of the stream type.
        int unique_stream_id = stream_profile.unique_id(); // The unique identifier can be used for comparing two streams
        std::cout << std::setw(3) << profile_num << ": " << stream_data_type << " #" << stream_index;

        // As noted, a stream is an abstraction.
        // In order to get additional data for the specific type of a
        //  stream, a mechanism of "Is" and "As" is provided:
        if (stream_profile.is<rs2::video_stream_profile>()) //"Is" will test if the type tested is of the type given
        {
            // "As" will try to convert the instance to the given type
            rs2::video_stream_profile video_stream_profile = stream_profile.as<rs2::video_stream_profile>();

            // After using the "as" method we can use the new data type
            //  for additinal operations:
            std::cout << " (Video Stream: " << video_stream_profile.format() << " " <<
                video_stream_profile.width() << "x" << video_stream_profile.height() << "@ " << video_stream_profile.fps() << "Hz)";
        }
        std::cout << std::endl;
        profile_num++;
    }

    uint32_t selected_profile_index = get_user_selection("Please select the desired streaming profile: ");
    if (selected_profile_index >= stream_profiles.size())
    {
        throw std::out_of_range("Requested profile index is out of range");
    }

    return stream_profiles[selected_profile_index];
}

void get_field_of_view(const rs2::stream_profile& stream)
{
    // A sensor's stream (rs2::stream_profile) is in general a stream of data with no specific type.
    // For video streams (streams of images), the sensor that produces the data has a lens and thus has properties such
    //  as a focal point, distortion, and principal point.
    // To get these intrinsics parameters, we need to take a stream and first check if it is a video stream
    if (auto video_stream = stream.as<rs2::video_stream_profile>())
    {
        try
        {
            //If the stream is indeed a video stream, we can now simply call get_intrinsics()
            rs2_intrinsics intrinsics = video_stream.get_intrinsics();

            float cx = intrinsics.ppx;
            float cy = intrinsics.ppy;
            float fx = intrinsics.fx;
            float fy = intrinsics.fy;

            rs2_distortion model = intrinsics.model;

            std::cout << "Principal Point         : " << cx << ", " << cy << std::endl;
            std::cout << "Focal Length            : " << fx << ", " << fy << std::endl;
            std::cout << "Distortion Model        : " << model << std::endl;
            std::cout << "Distortion Coefficients : [" << intrinsics.coeffs[0] << "," << intrinsics.coeffs[1] << "," <<
                intrinsics.coeffs[2] << "," << intrinsics.coeffs[3] << "," << intrinsics.coeffs[4] << "]" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Failed to get intrinsics for the given stream. " << e.what() << std::endl;
        }
    }
    else if (auto motion_stream = stream.as<rs2::motion_stream_profile>())
    {
        try
        {
            //If the stream is indeed a motion stream, we can now simply call get_motion_intrinsics()
            rs2_motion_device_intrinsic intrinsics = motion_stream.get_motion_intrinsics();

            std::cout << " Scale X      cross axis      cross axis  Bias X \n";
            std::cout << " cross axis    Scale Y        cross axis  Bias Y  \n";
            std::cout << " cross axis    cross axis     Scale Z     Bias Z  \n";
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    std::cout << intrinsics.data[i][j] << "    ";
                }
                std::cout << "\n";
            }
            
            std::cout << "Variance of noise for X, Y, Z axis \n";
            for (int i = 0; i < 3; i++)
                std::cout << intrinsics.noise_variances[i] << " ";
            std::cout << "\n";

            std::cout << "Variance of bias for X, Y, Z axis \n";
            for (int i = 0; i < 3; i++)
                std::cout << intrinsics.bias_variances[i] << " ";
            std::cout << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "Failed to get intrinsics for the given stream. " << e.what() << std::endl;
        }
    }
    else
    {
        std::cerr << "Given stream profile has no intrinsics data" << std::endl;
    }
}
