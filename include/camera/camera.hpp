#ifndef CAMERA_HPP
#define CAMERA_HPP

uint32_t get_user_selection(const std::string& prompt_message);
std::string get_sensor_name(const rs2::sensor& sensor);
rs2::sensor get_a_sensor_from_a_device(const rs2::device& dev);
rs2::stream_profile choose_a_streaming_profile(const rs2::sensor& sensor);
void get_field_of_view(const rs2::stream_profile& stream);
void get_extrinsics(const rs2::stream_profile& from_stream, const rs2::stream_profile& to_stream);
void show_extrinsics_between_streams(rs2::device device, rs2::sensor sensor);
std::string formatTimeStamp(double timestamp_ms);
#endif