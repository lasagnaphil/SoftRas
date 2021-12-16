#pragma once

#include <torch/torch.h>

namespace softras {

at::Tensor create_texture_image(
        at::Tensor vertices_all,
        at::Tensor textures,
        at::Tensor image,
        float eps);

at::Tensor load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update);

std::vector<at::Tensor> forward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_width,
        int image_height,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);

std::vector<at::Tensor> backward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_width,
        int image_height,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);

std::vector<at::Tensor> voxelize_sub1(
        at::Tensor faces,
        at::Tensor voxels);

std::vector<at::Tensor> voxelize_sub2(
        at::Tensor faces,
        at::Tensor voxels);

std::vector<at::Tensor> voxelize_sub3(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);

std::vector<at::Tensor> voxelize_sub4(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);

enum class DistFunc {
    Hard, Barycentric, Euclidean
};

enum class RgbFunc {
    Hard, SoftMax
};

enum class AlphaFunc {
    Hard, Sum, Prod
};

enum class TextureType {
    Surface, Vertex
};

class SoftRasterizeFunction : public torch::autograd::Function<SoftRasterizeFunction> {
public:
    static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor face_vertices, torch::Tensor textures, std::array<int, 2> image_size = {256, 256},
            std::array<float, 3> background_color = {0, 0, 0},
            float near = 1, float far = 100,
            bool fill_back = true, float eps = 1e-3,
            float sigma_val = 1e-5, DistFunc dist_func = DistFunc::Euclidean, float dist_eps = 1e-4,
            float gamma_val = 1e-4, RgbFunc aggr_func_rgb = RgbFunc::SoftMax, AlphaFunc aggr_func_alpha = AlphaFunc::Prod,
            TextureType texture_type = TextureType::Surface);

    static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
};

}