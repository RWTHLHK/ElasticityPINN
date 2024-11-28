import torch
import pytest
from Geometry import BoundaryUpdater, subdivide_bound_regions

def test_boundary_updater():
    """
    Test function for the BoundaryUpdater class.
    """
    # Define a test surface function: unit circle f(x) = x^2 + y^2 - 1
    def surface_func(points):
        radius = 1.0
        return torch.sum(points**2, dim=-1) - radius**2

    # Create an instance of the BoundaryUpdater class
    boundary_updater = BoundaryUpdater(surface_func)

    # Test Case 1: Normal behavior with a valid displacement field
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.707, 0.707]], requires_grad=True)
    u = torch.tensor([[0.1, 0.1], [0.0, 0.0], [-0.1, -0.1], [0.1, -0.1]], requires_grad=True)
    deformed_boundary = boundary_updater.update(u)
    result = deformed_boundary(x)
    expected = surface_func(x - u)

    assert torch.allclose(result, expected), "Test Case 1 Failed: Result does not match expected values."

    # Test Case 2: Ensure it raises an error for mismatched shapes
    try:
        x_invalid = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # Fewer points than in u
        deformed_boundary(x_invalid)
        assert False, "Test Case 2 Failed: Did not raise an error for mismatched shapes."
    except ValueError as e:
        assert "Shape of input x must match the shape of displacement u." in str(e), \
            "Test Case 2 Failed: Incorrect error message."

    # Test Case 3: Ensure it handles constant displacement (broadcasting)
    u_constant = torch.tensor([0.1, -0.1], requires_grad=True)  # Single displacement for all points
    deformed_boundary_constant = boundary_updater.update(u_constant.unsqueeze(0).expand_as(x))
    result_constant = deformed_boundary_constant(x)
    expected_constant = surface_func(x - u_constant.unsqueeze(0).expand_as(x))

    assert torch.allclose(result_constant, expected_constant), \
        "Test Case 3 Failed: Result does not match expected values for constant displacement."

    # Test Case 4: Ensure it handles gradients correctly
    result.sum().backward()  # Compute gradients for x and u
    assert x.grad is not None, "Test Case 4 Failed: Gradients for x were not computed."
    assert u.grad is not None, "Test Case 4 Failed: Gradients for u were not computed."

    print("All test cases passed successfully!")

def test_subdivide_bound_regions():
    """
    Test function for subdivide_bound_regions.
    """
    # Define 2D bounds: [x_min, x_max] and [y_min, y_max]
    bounds_2d = torch.tensor([[-1.5, 1.5], [-1.5, 1.5]])
    divisions_2d = [2, 2]  # Divide into a 2x2 grid

    # Call the function
    subregions_2d = subdivide_bound_regions(bounds_2d, divisions_2d, device='cpu', debug=True)

    # Expected subregions (manually calculated)
    expected_2d = torch.tensor([
        [[-1.5, 0.0], [-1.5,  0.0]],
        [[ 0.0, 1.5], [-1.5,  0.0]],
        [[-1.5,  0.0], [0.0,  1.5]],
        [[ 0.0,  1.5], [0,  1.5]],
    ])

    assert subregions_2d.shape == expected_2d.shape, "2D Test failed mismatch numbers of subregions"
    # Check that every expected subregion exists in the generated subregions
    for region in expected_2d:
        match_found = False
        for generated_region in subregions_2d:
            if torch.allclose(region, generated_region):
                match_found = True
                break
        assert match_found, f"Expected region {region} not found in generated regions."

    # Define 3D bounds: [x_min, x_max], [y_min, y_max], [z_min, z_max]
    bounds_3d = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    divisions_3d = [2, 2, 2]  # Divide into a 2x2x2 grid

    # Call the function
    subregions_3d = subdivide_bound_regions(bounds_3d, divisions_3d, device='cpu', debug=True)

    # Expected number of subregions
    expected_num_3d = 2 * 2 * 2  # 8 subregions

    # Validate the shape of the output
    assert subregions_3d.shape[0] == expected_num_3d, "Test failed for 3D case: incorrect number of subregions."
    assert subregions_3d.shape[1] == 3, "Test failed for 3D case: incorrect dimension of subregions."
    assert subregions_3d.shape[2] == 2, "Test failed for 3D case: incorrect bounds format."

# Run the test
if __name__ == "__main__":
    test_subdivide_bound_regions()
    # pytest.main()

