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

    virtual void Initialize() = 0;
    virtual void Export(const float4* positions, const float4* accelerations, const float* pressure, const float* density, uint32_t step, float dt) = 0;
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

        void Initialize() override {
            std::filesystem::path full_path = outpath_ / filename_;
            std::filesystem::create_directories(outpath_);

            outfile_.open(outpath_ / filename_);
            if (!outfile_.is_open()) { throw std::runtime_error("Failed to open CSV file at " + full_path.string()); }

            outfile_ << "Time,ID,PosX,PosY,PosZ,AccelX,AccelY,AccelZ,Mass,Density,Pressure\n";
        }

        void Export(const float4* positions, const float4* accelerations, const float* pressure, const float* density, uint32_t step, float dt) override {

            for (int i = 0; i < scene_settings.particle_count; i++) {
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
}

#endif //SPH_EXPORTERS_H
