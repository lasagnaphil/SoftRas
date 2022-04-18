#pragma once

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
        int image_size,
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
        int image_size,
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
