import os
import argparse
import numpy as np
import torch
import math
import json
import torch.nn as nn
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


# Add other necessary imports here

_EPS = 0.1


class Gaussians2D(nn.Module):
    def __init__(self, number_of_gaussians, mean, depth, radii, device):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, device=device), requires_grad=False)
        print(self.mean)
        self.color = torch.tensor([[255, 0, 0]] * number_of_gaussians, device=device)
        print(self.color)
        self.depth = nn.Parameter(
            (torch.tensor(depth, device=device) / 2.5 + 0.6).clamp(_EPS, 1) * 10,
            requires_grad=True,
        )
        print(self.depth)
        self.radii = nn.Parameter(
            torch.abs(torch.tensor(radii.float() / 1000, device=device)),
            requires_grad=True,
        )
        print(self.radii)
        self.number_of_gaussians = number_of_gaussians

    def forward(
        self,
    ):
        # Create grid
        xx = torch.linspace(0, 1023, 1024, device=self.depth.device)
        yy = torch.linspace(0, 1023, 1024, device=self.depth.device)
        xx, yy = torch.meshgrid(xx, yy)

        log_std = self.radii

        final_tensor = torch.zeros(
            3, 1024, 1024, device=self.depth.device
        )  ##############SARI: when allocating new memory, make sure it is on the same device (gpu/cpu) as all other memory

        for i in range(500):
            if self.mean == [0.0, 0.0]:
                i -= 1
                continue

            xx_rot = (xx - self.mean[i][0]) - (yy - self.mean[i][1]) + self.mean[i][0]
            yy_rot = (xx - self.mean[i][0]) + (yy - self.mean[i][1]) + self.mean[i][1]

            # Calculate Gaussian
            gaussian = torch.exp(
                -0.5
                * (
                    (xx_rot - self.mean[i][0]) ** 2 / torch.exp(2 * log_std[i])
                    + (yy_rot - self.mean[i][1]) ** 2 / torch.exp(2 * log_std[i])
                )
            )
            ############## SARI: you forgot the sqrt (I divided by 2 in the exponent which is equivilent), this means that you didn't normalize the gaussians correctly
            gaussian = (
                self.depth[i]
                * gaussian
                / (2 * np.pi * torch.exp(log_std[i] / 2) * torch.exp(log_std[i] / 2))
            )
            # Scale Gaussian by color
            colored_gaussian = gaussian.unsqueeze(0).repeat(
                3, 1, 1
            )  # Repeat Gaussian for 3 channels
            colored_gaussian *= self.color[i].view(3, 1, 1)  # Scale Gaussian by color

            final_tensor += colored_gaussian

        return final_tensor


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
    image_array = tensor.cpu().numpy()

    # Reshape the array if necessary
    if len(image_array.shape) == 4:
        image_array = np.squeeze(image_array, axis=0)

    # Plot the image with grid
    plt.imshow(image_array, extent=[-1, 1, -1, 1])
    plt.grid(True, color="gray", linestyle="-", linewidth=0.5)  # Show grid
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
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


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

    print(viewmat_np)
    viewmat = torch.tensor(
        np.matmul(np.transpose(viewmat_np), rotate_x(-math.pi / 2, device=device)),
        device=device,
    ).float()

    print(viewmat)

    fx = 35.0  # cam.data.lens = 35
    fy = 35.0
    cx = 16.0  # cx  - cam.data.sensor_width = 32
    cy = 16.0  # cy

    img_height = 1024  # find a way to input non-manually
    img_width = 1024  # find a way to input non-manually
    block_width = 16
    clip_thresh = 0.01

    fovx = 2 * math.atan(img_width / (2 * fx))
    fovy = 2 * math.atan(img_height / (2 * fy))
    projmat = projection_matrix(0.5, 100, fovx, fovy, device=device)

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

    # parameters
    load_image = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussians_to_train = Gaussians2D(
        number_of_gaussians=xys.shape[0],
        mean=xys,
        depth=depth,
        radii=radii,
        device=device,
    ).to(device)  #############SARI: moved the model(network) to the device
    output = gaussians_to_train()
    plt.imshow(
        output.permute(1, 2, 0).cpu().detach().numpy()
    )  # Permute dimensions for displaying RGB image
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
