#ifndef SPH_EXPORTERS_H
#define SPH_EXPORTERS_H

#include <filesystem>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <string>

#include "../scene/scene_settings.h"

struct Exporter {
    SceneSettings scene_settings;

    virtual void Initialize(uint32_t total_particles) = 0;
    virtual void Export(const float4* positions, const float4* accelerations, const float* pressure, const float* density, uint32_t step, float dt, uint32_t total_particles) = 0;
    virtual void Finalize() = 0;

    virtual ~Exporter() = default;
    Exporter(const SceneSettings &scene_settings) : scene_settings(scene_settings) {}
};

namespace Exporters {
    struct CSV : Exporter {
        std::ofstream outfile_;

        std::filesystem::path outpath_ = ".";
        std::filesystem::path filename_ = "sphdata";

        CSV(const SceneSettings &scene_settings, std::filesystem::path outpath, std::filesystem::path filename) : Exporter(scene_settings), outpath_(std::move(outpath)), filename_(std::move(filename)) {}

        void Initialize(uint32_t total_particles) override {
            std::filesystem::path full_path = outpath_ / filename_;
            std::filesystem::create_directories(outpath_);

            outfile_.open(outpath_ / filename_);
            if (!outfile_.is_open()) { throw std::runtime_error("Failed to open CSV file at " + full_path.string()); }

            outfile_ << "Time,ID,PosX,PosY,PosZ,AccelX,AccelY,AccelZ,Mass,Pressure,Density\n";
        }

        void Export(const float4* positions, const float4* accelerations, const float* pressure, const float* density, uint32_t step, float dt, uint32_t total_particles) override {

            for (int i = 0; i < total_particles; i++) {
                outfile_ << step * dt << ","
                    << i << ","
                    << positions[i].x << ","
                    << positions[i].y << ","
                    << positions[i].z << ","
                    << accelerations[i].x << ","
                    << accelerations[i].y << ","
                    << accelerations[i].z << ","
                    << positions[i].w << ","
                    << pressure[i] << ","
                    << density[i] << "\n";
            }
        }

        void Finalize() override {
            if (outfile_.is_open()) { outfile_.close(); }
        }
    };

    struct XDMF : Exporter {
        std::ofstream xmf_file_;
        std::ofstream bin_file_;
        std::filesystem::path bin_filename_ = "nbodydata_bin";
        std::filesystem::path xmf_filename_ = "nbodydata_xmf";
        std::filesystem::path outpath_ = ".";

        std::vector<float> pos_buf_;
        std::vector<float> accel_buf_;
        std::vector<float> prs_buf_;
        std::vector<float> den_buf_;

        XDMF(const SceneSettings &scene_settings, std::filesystem::path outpath, std::filesystem::path bin_filename, std::filesystem::path xmf_filename) : Exporter(scene_settings), bin_filename_(std::move(bin_filename)), xmf_filename_(std::move(xmf_filename)), outpath_(std::move(outpath)) {}

        void Initialize(uint32_t total_particles) override {
            std::filesystem::create_directories(outpath_);

            xmf_filename_.replace_extension(".xmf");
            bin_filename_.replace_extension(".bin");
            std::filesystem::path xmf_full_path = outpath_ / xmf_filename_;
            std::filesystem::path bin_full_path = outpath_ / bin_filename_;

            xmf_file_.open(xmf_full_path);
            if (!xmf_file_.is_open()) { throw std::runtime_error("Failed to open XDMF file at " + xmf_full_path.string()); }

            bin_file_.open(bin_full_path, std::ios::binary);
            if (!bin_file_.is_open()) { throw std::runtime_error("Failed to open binary file at " + bin_full_path.string()); }

            pos_buf_.resize(total_particles * 3);
            accel_buf_.resize(total_particles * 3);

            prs_buf_.resize(total_particles);
            den_buf_.resize(total_particles);

            xmf_file_ << R"(<?xml version="1.0" ?>
            <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
            <Xdmf Version="2.0">
              <Domain>
                <Grid Name="Simulation" GridType="Collection" CollectionType="Temporal">
            )";
        }

        void Export(const float4* positions, const float4* accelerations, const float* pressure, const float* density, uint32_t step, float dt, uint32_t total_particles) override {
            uint64_t vec_bytes = total_particles * 3 * sizeof(float);
            uint64_t scl_bytes = total_particles * sizeof(float);

            for (int i = 0; i < total_particles; ++i) {
                pos_buf_[i * 3 + 0] = positions[i].x;
                pos_buf_[i * 3 + 1] = positions[i].y;
                pos_buf_[i * 3 + 2] = positions[i].z;

                accel_buf_[i * 3 + 0] = accelerations[i].x;
                accel_buf_[i * 3 + 1] = accelerations[i].y;
                accel_buf_[i * 3 + 2] = accelerations[i].z;

                prs_buf_[i] = pressure[i];
                den_buf_[i] = density[i];
            }

            uint64_t step_start_offset = bin_file_.tellp();

            bin_file_.write(reinterpret_cast<const char*>(pos_buf_.data()), vec_bytes);
            bin_file_.write(reinterpret_cast<const char*>(accel_buf_.data()), vec_bytes);
            bin_file_.write(reinterpret_cast<const char*>(prs_buf_.data()), scl_bytes);
            bin_file_.write(reinterpret_cast<const char*>(den_buf_.data()), scl_bytes);
            bin_file_.flush();

            uint64_t pos_offset  = step_start_offset;
            uint64_t vel_offset  = step_start_offset + vec_bytes;
            uint64_t prs_offset = step_start_offset + vec_bytes + vec_bytes;
            uint64_t den_offset = step_start_offset + vec_bytes + vec_bytes + scl_bytes;

            xmf_file_ <<
                R"(<Grid Name="Step_)" << step << R"(" GridType="Uniform">
                    <Time Value=")" << step * dt << R"(" />
                    <Topology TopologyType="Polyvertex" NumberOfElements=")" << total_particles << R"("/>
                    <Geometry GeometryType="XYZ">
                      <DataItem Dimensions=")" << total_particles << R"( 3" NumberType="Float" Precision="4" Format="Binary" Seek=")" << pos_offset << R"(" Endian="Native">
                        )" << bin_filename_.string() << R"(
                      </DataItem>
                    </Geometry>
                    <Attribute Name="Acceleration" AttributeType="Vector" Center="Node">
                      <DataItem Dimensions=")" << total_particles << R"( 3" NumberType="Float" Precision="4" Format="Binary" Seek=")" << vel_offset << R"(" Endian="Native">
                        )" << bin_filename_.string() << R"(
                      </DataItem>
                    </Attribute>
                    <Attribute Name="Pressure" AttributeType="Scalar" Center="Node">
                      <DataItem Dimensions=")" << total_particles << R"(" NumberType="Float" Precision="4" Format="Binary" Seek=")" << prs_offset << R"(" Endian="Native">
                        )" << bin_filename_.string() << R"(
                      </DataItem>
                    </Attribute>
                    <Attribute Name="Density" AttributeType="Scalar" Center="Node">
                      <DataItem Dimensions=")" << total_particles << R"(" NumberType="Float" Precision="4" Format="Binary" Seek=")" << den_offset << R"(" Endian="Native">
                        )" << bin_filename_.string() << R"(
                      </DataItem>
                    </Attribute>
                  </Grid>
            )";

            xmf_file_.flush();
        }

        void Finalize() override {
            if (bin_file_.is_open()) { bin_file_.close(); }

            if (xmf_file_.is_open()) {
                xmf_file_ <<
                    R"(    </Grid>
                      </Domain>
                    </Xdmf>
                    )";
                xmf_file_.close();
            }
        }
    };


}

#endif //SPH_EXPORTERS_H
