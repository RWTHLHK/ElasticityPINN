import torch
from Sample import mcmc_sampler, plot_samples_with_boundary_2d 
from Geometry import subdivide_bound_regions

if __name__ == "__main__":
    # Define a 2D boundary function (unit circle)
    def boundary_func_2d(points):
        return torch.sum(points**2, dim=-1) - 1  # x^2 + y^2 - 1 <= 0

    # Define bounds and subregions
    bounds_2d = torch.tensor([[-1.5, 1.5], [-1.5, 1.5]])
    divisions_2d = [10, 10]
    subregions_2d = subdivide_bound_regions(bounds_2d, divisions_2d, device='cpu')

    # Perform weighted MCMC sampling
    num_samples_per_region = 10
    samples_2d = mcmc_sampler(boundary_func_2d, subregions_2d, num_samples_per_region, step_size=0.05, device='cuda')

    # Plot the samples with the boundary
    plot_samples_with_boundary_2d(samples_2d, boundary_func_2d, bounds_2d)

