import os
import argparse
import numpy as np
import torch
import math
import json
import viser.transforms as vtf
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


# Add other necessary imports here


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for extracting points from an object file."
    )
    parser.add_argument(
        "object_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/in2/03790512/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/model_normalized.obj",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out2/params/out.txt",
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/img/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/transforms.json",
    )
    """
    parser.add_argument(
        "out_dir",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out2/params",
    )
    parser.add_argument(
        "rotation_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/camera/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/rotation.npy",
    )
    parser.add_argument(
        "elevation_file",
        type=str,
        help="/mnt/d/Users/christeen-shaheen/temp/out/camera/1a48f3b3c0ab0fdeba8e0a95fbfb7c9/models/elevation.npy",
    )
    """
    # Add other arguments as needed
    return parser.parse_args()


def extract_points_from_obj(input_file):
    points = []
    counter = 0
    try:
        with open(input_file, "r") as f:
            for line in f:
                if line.startswith("v "):
                    counter += 1
                    # Extracting vertices (points)
                    parts = line.split()
                    # Assuming format: v x y z
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
    except Exception as e:
        print(f"Error extracting points from {input_file}: {e}")

    return points


def save_points_to_file(points_tensor, output_file):
    try:
        with open(output_file, "w") as f:
            for point in points_tensor:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"Points saved to {output_file} successfully.")
    except Exception as e:
        print(f"Error saving points to {output_file}: {e}")


def extract_transform_matrix(file_path, index):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract the transform matrix at the specified index
    if 0 <= index < len(data["frames"]):
        transform_matrix = data["frames"][index]["transform_matrix"]
        return transform_matrix
    else:
        print(f"Index {index} is out of range.")
        return None


def show_image_from_tensor(tensor):
    # Convert the tensor to a NumPy array
    # print(tensor)
    image_array = tensor.cpu().numpy()

    # Reshape the array if necessary
    # if len(image_array.shape) == 4:
    #     image_array = np.squeeze(image_array, axis=0)

    # Plot the image
    plt.imshow(image_array)
    # print(image_array)
    plt.axis("on")  # Turn off axis
    plt.show()


def tensor_to_image(tensor, range_min=-1024, range_max=1024):
    # Convert the tensor to a NumPy array
    image_array = tensor.cpu().numpy()

    # Scale the values to fit within the desired range
    scaled_image_array = (
        (image_array - image_array.min()) / (image_array.max() - image_array.min())
    ) * (range_max - range_min) + range_min

    # Plot the image
    plt.imshow(
        scaled_image_array,
        cmap="gray",
        extent=[range_min, range_max, range_min, range_max],
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Image from Tensor")
    plt.colorbar()  # Add colorbar for reference
    plt.show()


def visualize_3d_image(image_tensor):
    # Convert tensor to numpy array
    image_array = image_tensor.numpy()

    # Get dimensions of the image
    depth, height, width = image_array.shape

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each voxel in the image
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if image_array[z, y, x] > 0:  # Only plot non-zero voxels
                    ax.scatter(x, y, -z, c="b", marker="o")

    # Set labels and invert z-axis to match usual image convention
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(depth, 0)
    plt.show()


def print_tensor_as_gif(tensor, out_dir):
    frames = []
    frames.append((tensor.detach().cpu().numpy() * 255).astype(np.uint8))
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(
        f"{out_dir}/training.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=5,
        loop=0,
    )


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    ).float()


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return np.array(([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]))


def main():
    # Extract points from the object file
    device = torch.device("cuda:0")
    args = parse_arguments()
    points_tensor = torch.tensor(
        extract_points_from_obj(args.object_file), device=device
    )
    save_points_to_file(points_tensor, args.output_file)

    u = points_tensor[:, 0]
    v = points_tensor[:, 1]
    w = points_tensor[:, 2]

    means3d = torch.tensor(points_tensor, device=device)
    inner_values = [0.9, 0.7, 0.9]
    scales = torch.tensor([inner_values] * means3d.shape[0], device=device)
    glob_scale = 0.9
    quats = torch.cat(
        [
            torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
            torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
            torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
            torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        ],
        -1,
    )
    quats = torch.tensor(quats, device=device)
    viewmat_np = np.array(extract_transform_matrix(args.json_file, 0))
    # Tried the r and t matrices like them for the view matrix
    """
    # Extract the first 3 rows and columns and put them in R matrix
    R = viewmat_np[:3, :3]

    # Extract the last column and put it in T matrix
    T = viewmat_np[:, -1].reshape(-1, 1)  # Reshape to make T a column vector

    # flip the z axis to align with gsplat conventions
    R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=device)
    R = torch.tensor(R, device=device) @ R_edit

    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ torch.tensor(T, device=device)

    viewmat = torch.eye(4, device=device).float()
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    # viewmat = torch.tensor(viewmat_np, device=device)
    # viewmat = viewmat @ rotate_x(-math.pi / 2, device=device)
    """

    viewmat = torch.tensor(
        np.matmul(np.transpose(viewmat_np), rotate_x(-math.pi / 2, device=device)),
        device=device,
    ).float()

    fx = 0.5 * float(1024) / math.tan(0.5 * math.pi / 2.0)  # cam.data.lens = 35
    fy = 0.5 * float(1024) / math.tan(0.5 * math.pi / 2.0)
    cx = 512.0  # cx  - cam.data.sensor_width = 32
    cy = 512.0  # cy

    # camera_to_pixel = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
    # projmat = torch.tensor(
    #     np.matmul(camera_to_pixel, viewmat_np), device=device
    # ).float()

    img_height = 1024  # find a way to input non-manually
    img_width = 1024  # find a way to input non-manually
    block_width = 16
    clip_thresh = 0.01

    fovx = 2 * math.atan(img_width / (2 * fx))
    fovy = 2 * math.atan(img_height / (2 * fy))
    projmat = projection_matrix(0.001, 1000, fovx, fovy, device=device)
    projmat = projmat.squeeze() @ viewmat.squeeze()

    out_project = project_gaussians(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )

    xys, depth, radii, conics, compensation, num_tiles_hit, cov3d = out_project
    # rgbs = torch.rand(means3d.shape[0], 3, device=device)
    rgbs = torch.tensor([[0, 0, 255]] * means3d.shape[0], device=device).float()
    opacities = torch.ones((means3d.shape[0], 1), device=device)

    final_out = rasterize_gaussians(
        xys,
        depth,
        radii // 10000,
        conics,
        num_tiles_hit,
        (rgbs),
        torch.sigmoid(opacities),
        img_height,
        img_width,
        block_width,
    )

    """
    if not torch.equal(means3d, points_tensor):
        print("chnaged")

    
    for i in range(100):
        out_project = project_gaussians(
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            projmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        xys, depth, radii, conics, compensation, num_tiles_hit, cov3d = out_project

        final_out = rasterize_gaussians(
            xys,
            depth,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(rgbs),
            torch.sigmoid(opacities),
            img_height,
            img_width,
            block_width,
        )
    """
    # show_image_from_tensor(final_out)
    tensor_to_image(final_out)


if __name__ == "__main__":
    main()
