import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Debugger import *
from typing import Union

class BoundaryUpdater:
    """
    A class to define and update the boundary of a domain based on an initial boundary function f(x).
    """
    def __init__(self, f: callable):
        """
        Initialize the BoundaryUpdater with the boundary function.

        Parameters:
        f (callable): The initial boundary function f(x), defining the surface condition.
        """
        self.f = f  # Fixed boundary function

    def update(self, u: torch.Tensor):
        """
        Update the boundary by specifying a displacement field u.

        Parameters:
        u (torch.Tensor): Displacement field (N, D), where N is the number of points and D is the dimensionality.

        Returns:
        callable: A function that computes f(x - u) for a given x.
        """

        # Return a callable function that computes f(x - u)
        def deformed_boundary(x: torch.Tensor) -> torch.Tensor:
            if x.shape != u.shape:
                raise ValueError("Shape of input x must match the shape of displacement u.")
            # Compute deformed coordinates
            x_deformed = x - u
            # Compute and return boundary condition f(x - u)
            return self.f(x_deformed)
        
        return deformed_boundary

@debug_mode
def subdivide_bound_regions(bounds:torch.Tensor, divisions:Union[list, tuple], device:str='cuda', debug=False):
    """
    Subdivide a 2D or 3D bound region into smaller subregions.

    Parameters:
    - bounds (torch.Tensor): A tensor of shape (dim, 2) specifying the lower and upper bounds for each dimension.
    - divisions (list or tuple): Number of divisions along each dimension (e.g., [nx, ny] for 2D or [nx, ny, nz] for 3D).
    - device (str): Device to perform the computation ('cuda' for GPU, 'cpu' for CPU).

    Returns:
    - subregions (torch.Tensor): A tensor of shape (num_subregions, dim, 2), where each entry contains the
      lower and upper bounds of a subregion.
    """
    dim = bounds.shape[0]  # Dimensionality of the space
    if len(divisions) != dim:
        logger.error(f"Divisions length {len(divisions)} does not match bounds dimensionality {dim}.")
        raise ValueError(f"Divisions must match the dimensionality of bounds. Expected {dim}, got {len(divisions)}.")
    
    # Log input parameters
    logger.debug(f"Bounds: {bounds}")
    logger.debug(f"Divisions: {divisions}")
    logger.debug(f"Device: {device}")
    logger.debug(f"dim: {dim}")

    # Generate grid edges along each dimension
    grid_edges = [
        torch.linspace(bounds[i, 0], bounds[i, 1], divisions[i] + 1, device=device)
        for i in range(dim)
    ]
    logger.debug(f"dim of grid_edges: {len(grid_edges)}")
    logger.debug(f"grid_edges are: ")
    logger.debug(grid_edges)
    # Create meshgrid for subregion vertices
    grids = torch.meshgrid(*grid_edges, indexing="ij")  # Grids for each dimension
    logger.debug(f"mesh grids are: ")
    logger.debug(grids)
    # Create meshgrid of indices
    mesh = torch.meshgrid(*[torch.arange(divisions[d], device=device) for d in range(dim)], indexing="ij")

    # Iterate over meshgrid indices to compute bounds for each subregion
    subregions = []
    for indices in zip(*(m.flatten() for m in mesh)):
        subregion = []
        for d, idx in enumerate(indices):
            lower = grid_edges[d][idx]
            upper = grid_edges[d][idx + 1]
            subregion.append([lower.item(), upper.item()])
        subregions.append(subregion)

    # Convert subregions to a tensor
    subregions = torch.tensor(subregions, device=device)

    return subregions

def plot_subregions(subregions):
    """
    Plot subregions in 2D or 3D based on the dimensionality of the input.

    Parameters:
    - subregions (torch.Tensor): A tensor of shape (num_subregions, dim, 2) containing bounds for each subregion.
    """
    dim = subregions.shape[1]  # Dimensionality of the region (2D or 3D)

    if dim == 2:
        # 2D Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        for region in subregions:
            lower = region[:, 0].cpu().numpy()  # Lower bounds [x_min, y_min]
            upper = region[:, 1].cpu().numpy()  # Upper bounds [x_max, y_max]

            # Rectangle parameters
            width, height = upper[0] - lower[0], upper[1] - lower[1]
            rect = plt.Rectangle(lower, width, height, color='blue', alpha=0.5, edgecolor='black')
            ax.add_patch(rect)

        ax.set_xlim(subregions[:, 0, 0].min().cpu().numpy(), subregions[:, 0, 1].max().cpu().numpy())
        ax.set_ylim(subregions[:, 1, 0].min().cpu().numpy(), subregions[:, 1, 1].max().cpu().numpy())
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.title('2D Subregions')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    elif dim == 3:
        # 3D Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for region in subregions:
            lower = region[:, 0].cpu().numpy()  # Lower bounds [x_min, y_min, z_min]
            upper = region[:, 1].cpu().numpy()  # Upper bounds [x_max, y_max, z_max]

            # Create the vertices of the cube
            vertices = [
                [lower[0], lower[1], lower[2]],
                [lower[0], lower[1], upper[2]],
                [lower[0], upper[1], lower[2]],
                [lower[0], upper[1], upper[2]],
                [upper[0], lower[1], lower[2]],
                [upper[0], lower[1], upper[2]],
                [upper[0], upper[1], lower[2]],
                [upper[0], upper[1], upper[2]],
            ]

            # Define the 12 edges of the cube
            edges = [
                [vertices[0], vertices[1], vertices[3], vertices[2]],  # Bottom face
                [vertices[4], vertices[5], vertices[7], vertices[6]],  # Top face
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face
                [vertices[0], vertices[2], vertices[6], vertices[4]],  # Side face
                [vertices[1], vertices[3], vertices[7], vertices[5]],  # Side face
            ]

            # Add the edges to the plot
            poly = Poly3DCollection(edges, alpha=0.3, edgecolor='k')
            ax.add_collection3d(poly)

        # Set axis limits
        ax.set_xlim(subregions[:, 0, 0].min().cpu().numpy(), subregions[:, 0, 1].max().cpu().numpy())
        ax.set_ylim(subregions[:, 1, 0].min().cpu().numpy(), subregions[:, 1, 1].max().cpu().numpy())
        ax.set_zlim(subregions[:, 2, 0].min().cpu().numpy(), subregions[:, 2, 1].max().cpu().numpy())
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title('3D Subregions')
        plt.show()

    else:
        raise ValueError("Only 2D and 3D subregions are supported.")
    
# example usage
if __name__ == "__main__":
    # initial surface f(x): x^2 + y^2 - R^2 = 0 
    def surface_func(points):
        radius = 1.0
        return torch.sum(points**2, dim=-1) - radius**2

    # initilize boundary updater
    boundary_updater = BoundaryUpdater(surface_func)
    # define displacement field
    u = torch.tensor([
        [0.1, 0.0],
        [0.0, 0.2],
        [-0.1, -0.2],
        [0.2, 0.1]
    ], requires_grad=True)  # 
    # update deformed boundary
    deformed_boundary = boundary_updater.update(u)

    # coordinates
    x = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], requires_grad=True)  # surface boundary (N, 2)

    updated_boundary_values = deformed_boundary(x)

    # 输出结果
    print("Updated boundary values f(x - u):")
    print(updated_boundary_values)  # f(x - u)

    bounds_2d = torch.tensor([[-1, 1], [-1, 1]])
    divisions = (2, 2)
    sub_bound_regions = subdivide_bound_regions(bounds=bounds_2d, divisions=divisions, device='cpu')
    plot_subregions(sub_bound_regions)
