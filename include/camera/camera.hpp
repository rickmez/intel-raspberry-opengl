#ifndef CAMERA_HPP
#define CAMERA_HPP

uint32_t get_user_selection(const std::string& prompt_message);
std::string get_sensor_name(const rs2::sensor& sensor);
rs2::sensor get_a_sensor_from_a_device(const rs2::device& dev);
rs2::stream_profile choose_a_streaming_profile(const rs2::sensor& sensor);
void get_field_of_view(const rs2::stream_profile& stream);
#endif