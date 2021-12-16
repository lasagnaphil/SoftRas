//
// Created by lasagnaphil on 21. 12. 13..
//

#include "softras/api.h"

using namespace torch::autograd;
using namespace torch::indexing;
using torch::Tensor;

Tensor
softras::SoftRasterizeFunction::forward(AutogradContext* ctx, torch::Tensor face_vertices,
                                        torch::Tensor textures, std::array<int, 2> image_size,
                                        std::array<float, 3> background_color,
                                        float near, float far,
                                        bool fill_back, float eps, float sigma_val, DistFunc dist_func,
                                        float dist_eps, float gamma_val, RgbFunc aggr_func_rgb,
                                        AlphaFunc aggr_func_alpha, TextureType texture_type) {
    ctx->saved_data["image_width"] = image_size[0];
    ctx->saved_data["image_height"] = image_size[1];
    ctx->saved_data["background_color_r"] = background_color[0];
    ctx->saved_data["background_color_g"] = background_color[1];
    ctx->saved_data["background_color_b"] = background_color[2];
    ctx->saved_data["near"] = near;
    ctx->saved_data["far"] = far;
    ctx->saved_data["eps"] = eps;
    ctx->saved_data["sigma_val"] = sigma_val;
    ctx->saved_data["gamma_val"] = gamma_val;
    ctx->saved_data["func_dist_type"] = (int)dist_func;
    ctx->saved_data["dist_eps"] = logf(1.f / dist_eps - 1.f);
    ctx->saved_data["func_rgb_type"] = (int)aggr_func_rgb;
    ctx->saved_data["func_alpha_type"] = (int)aggr_func_alpha;
    ctx->saved_data["texture_type"] = (int)texture_type;
    ctx->saved_data["fill_back"] = fill_back;

    auto face_vertices_clone = face_vertices.clone();
    auto textures_clone = textures.clone();

    auto device = face_vertices_clone.device();
    int64_t batch_size = face_vertices_clone.size(0);
    int64_t num_faces = face_vertices_clone.size(1);

    ctx->saved_data["device"] = device;
    ctx->saved_data["batch_size"] = batch_size;
    ctx->saved_data["num_faces"] = num_faces;

    auto faces_info = torch::zeros({batch_size, num_faces, 9*3},
                                   torch::TensorOptions(torch::kFloat32).device(device));
    auto aggrs_info = torch::zeros({batch_size, 2, (int64_t)image_size[1], (int64_t)image_size[0]},
                                   torch::TensorOptions(torch::kFloat32).device(device));
    auto soft_colors = torch::ones({batch_size, 4, (int64_t)image_size[1], (int64_t)image_size[0]},
                                   torch::TensorOptions(torch::kFloat32).device(device));

    soft_colors.index({Slice(), 0, Slice(), Slice()}) *= background_color[0];
    soft_colors.index({Slice(), 1, Slice(), Slice()}) *= background_color[1];
    soft_colors.index({Slice(), 2, Slice(), Slice()}) *= background_color[2];

    auto res = forward_soft_rasterize(face_vertices, textures, faces_info, aggrs_info, soft_colors,
                                      image_size[0], image_size[1], near, far, eps, sigma_val, (int)dist_func, dist_eps,
                                      gamma_val, (int)aggr_func_rgb, (int)aggr_func_alpha,
                                      (int)texture_type, fill_back);

    faces_info = res[0];
    aggrs_info = res[1];
    soft_colors = res[2];
    ctx->save_for_backward({face_vertices, textures, soft_colors, faces_info, aggrs_info});

    return soft_colors;
}

tensor_list softras::SoftRasterizeFunction::backward(AutogradContext* ctx,
                                                     tensor_list grad_outputs) {

    auto saved_vars = ctx->get_saved_variables();
    auto face_vertices = saved_vars[0];
    auto textures = saved_vars[1];
    auto soft_colors = saved_vars[2];
    auto faces_info = saved_vars[3];
    auto aggrs_info = saved_vars[4];

    std::array<float, 3> background_color;
    background_color[0] = ctx->saved_data["background_color_r"].to<float>();
    background_color[1] = ctx->saved_data["background_color_g"].to<float>();
    background_color[2] = ctx->saved_data["background_color_b"].to<float>();

#define LOADVAR(name, type) auto (name) = ctx->saved_data[#name].to<type>();
    LOADVAR(image_width, int)
    LOADVAR(image_height, int)
    LOADVAR(near, float)
    LOADVAR(far, float)
    LOADVAR(eps, float)
    LOADVAR(sigma_val, float)
    LOADVAR(dist_eps, float)
    LOADVAR(gamma_val, float)
    LOADVAR(func_dist_type, int)
    LOADVAR(func_rgb_type, int)
    LOADVAR(func_alpha_type, int)
    LOADVAR(texture_type, int)
    LOADVAR(fill_back, bool)
#undef LOADVAR

    auto grad_faces = torch::zeros_like(saved_vars[0]);
    auto grad_textures = torch::zeros_like(saved_vars[1]);
    auto grad_soft_colors = saved_vars[2].contiguous();

    auto res = backward_soft_rasterize(face_vertices, textures, soft_colors, faces_info, aggrs_info,
                                       grad_faces, grad_textures, grad_soft_colors,
                                       image_width, image_height, near, far, eps,
                                       sigma_val, func_dist_type, dist_eps,
                                       gamma_val, func_rgb_type, func_alpha_type,
                                       texture_type, fill_back);

    return {grad_faces, grad_textures,
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
            Tensor()};
}
