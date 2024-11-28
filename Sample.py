import torch 
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Debugger import debug_mode, logger

@debug_mode
def mcmc_poisson_disk_sampler(boundary_func:callable, 
                              subregions:torch.Tensor,
                              min_distance:float, step_size:float=torch.nan, 
                              max_iters:int=10000, device='cuda', debug:bool=False):
    """
    Perform MCMC sampling with a Poisson Disk Sampling constraint and forbidden area around current points.

    Parameters:
    - boundary_func (callable): Function f(x), where f(x) <= 0 defines the boundary.
    - subregions (torch.Tensor): A tensor of shape (num_subregions, dim, 2) containing subregion bounds.
    - min_distance (float): Minimum distance between sampled points (Poisson Disk constraint).
    - step_size (float): Maximum distance for MCMC proposals.
    - max_iters (int): Maximum number of iterations for MCMC sampling.
    - device (str): 'cuda' for GPU or 'cpu' for CPU.

    Returns:
    - all_samples (torch.Tensor): A tensor of shape (N, dim) containing the sampled points.
    """
    if torch.isnan(step_size):
        step_size = 2*min_distance

    dim = subregions.shape[1]  # Dimensionality of the space

    def sample_in_subregion(region_bounds):
        """Perform Poisson Disk Sampling within a single subregion."""
        lower, upper = region_bounds[:, 0], region_bounds[:, 1]

        # Initialize random starting point
        current_point = torch.rand((1, dim), device=device) * (upper - lower) + lower
        samples = [current_point]

        for _ in range(max_iters):
            # Generate a new proposal that respects the forbidden area
            for _ in range(100):  # Limit the number of retries for generating a valid proposal
                # Random direction
                direction = torch.randn((1, dim), device=device)
                direction = direction / torch.norm(direction, dim=1, keepdim=True)  # Normalize to unit vector

                # Random distance within [min_distance, step_size]
                distance = torch.rand(1, device=device) * (step_size - min_distance) + min_distance

                # Generate a proposal in the spherical shell
                proposal = current_point + direction * distance

                # Check if the proposal is within bounds
                if not torch.all((proposal >= lower) & (proposal <= upper)):
                    continue

                # Check if the proposal satisfies the boundary condition
                if not (boundary_func(proposal) <= 0).item():
                    continue

                # Check distance to all existing points
                distances = torch.cdist(torch.cat(samples, dim=0), proposal)
                if torch.all(distances >= min_distance):
                    samples.append(proposal)
                    current_point = proposal  # Move to the new point
                    break

        # Concatenate all samples
        return torch.cat(samples, dim=0)

    # Sample for each subregion
    all_samples = []
    for region in subregions:
        region = region.to(device)
        samples = sample_in_subregion(region)
        all_samples.append(samples)

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    return all_samples


def mcmc_sampler(boundary_func:callable, 
                 subregions:torch.Tensor, 
                 num_samples_per_region:int, 
                 step_size:float=0.1, max_iters:int=10000, device:str='cuda'):
    """
    Perform MCMC sampling on precomputed subregions iteratively or in parallel.

    Parameters:
    - boundary_func (callable): Function f(x), where f(x) <= 0 defines the boundary.
    - subregions (torch.Tensor): A tensor of shape (num_subregions, dim, 2) containing subregion bounds.
    - num_samples_per_region (int): Number of samples to generate in each subregion.
    - step_size (float): Step size for the proposal distribution.
    - max_iters (int): Maximum number of iterations for MCMC sampling.
    - device (str): 'cuda' for GPU or 'cpu' for CPU.

    Returns:
    - all_samples (torch.Tensor): A tensor of shape (total_samples, dim) containing sampled points from all subregions.
    """
    dim = subregions.shape[1]  # Dimensionality of the space

    def sample_in_subregion(region_bounds):
        """
        Perform MCMC sampling within a single subregion.
        """
        lower, upper = region_bounds[:, 0], region_bounds[:, 1]

        # Initialize random starting points
        current_points = torch.rand((num_samples_per_region, dim), device=device) * (upper - lower) + lower
        samples = []

        for _ in range(max_iters):
            # Propose new points
            proposals = current_points + step_size * torch.randn_like(current_points, device=device)

            # Check bounds
            within_bounds = torch.all((proposals >= lower) & (proposals <= upper), dim=1)

            # Check boundary condition
            within_boundary = boundary_func(proposals) <= 0

            # Accept valid points
            accepted = within_bounds & within_boundary
            samples.append(proposals[accepted])

            # Update current points for accepted proposals
            current_points[accepted] = proposals[accepted]

            # Stop if we have enough samples
            total_samples = sum(s.shape[0] for s in samples)
            if total_samples >= num_samples_per_region:
                break

        # Concatenate samples and trim to the required number
        samples = torch.cat(samples, dim=0)
        if samples.shape[0] > num_samples_per_region:
            samples = samples[:num_samples_per_region]

        return samples

    # Perform sampling for each subregion (Parallel or Iterative)
    all_samples = []

    if device == 'cuda':
        # GPU-based sampling: Process all subregions in a single loop
        for region in subregions:
            region = region.to(device)
            samples = sample_in_subregion(region)
            all_samples.append(samples)
    else:
        # CPU-based sampling: Use ThreadPoolExecutor for parallelism
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(sample_in_subregion, region.to(device)) for region in subregions]
            all_samples = [future.result() for future in futures]

    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    return all_samples

def plot_samples_with_boundary_2d(samples, boundary_func, bounds, resolution=100):
    """
    Plot 2D sampled points along with the boundary.

    Parameters:
    - samples (torch.Tensor): Sampled points (N, 2).
    - boundary_func (callable): Boundary function f(x, y) <= 0.
    - bounds (torch.Tensor): Bounds of the region, shape (2, 2) -> [[x_min, x_max], [y_min, y_max]].
    - resolution (int): Number of points along each axis for the boundary plot.
    """
    x = torch.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = torch.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = boundary_func(torch.stack([X.flatten(), Y.flatten()], dim=-1)).reshape(X.shape)

    plt.figure(figsize=(8, 8))
    contour = plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=[0], colors='red', linewidths=2, linestyles='--')
    plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), s=10, alpha=0.7, label='Sampled Points')

    # Add legend manually
    custom_lines = [
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Boundary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Sampled Points')
    ]
    plt.legend(handles=custom_lines)

    plt.xlim(bounds[0, 0].item(), bounds[0, 1].item())
    plt.ylim(bounds[1, 0].item(), bounds[1, 1].item())
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Sampled Points with Boundary')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_samples_with_boundary_3d(samples, boundary_func, bounds, resolution=50):
    """
    Plot 3D sampled points along with the boundary.

    Parameters:
    - samples (torch.Tensor): Sampled points (N, 3).
    - boundary_func (callable): Boundary function f(x, y, z) <= 0.
    - bounds (torch.Tensor): Bounds of the region, shape (3, 2) -> [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
    - resolution (int): Number of points along each axis for the boundary plot.
    """
    x = torch.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = torch.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Solve for Z on the boundary (f(x, y, z) = 0)
    def solve_z(x, y):
        return torch.sqrt(1 - x**2 - y**2)  # Example: Sphere boundary (x^2 + y^2 + z^2 = 1)

    Z = solve_z(X, Y)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the boundary surface
    ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), color='red', alpha=0.5, label='Boundary')

    # Plot sampled points
    ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), samples[:, 2].numpy(), s=10, alpha=0.7, label='Sampled Points')

    ax.set_xlim(bounds[0, 0].item(), bounds[0, 1].item())
    ax.set_ylim(bounds[1, 0].item(), bounds[1, 1].item())
    ax.set_zlim(bounds[2, 0].item(), bounds[2, 1].item())
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('3D Sampled Points with Boundary')
    plt.legend()
    plt.show()
