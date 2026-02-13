import torch
import numpy as np

from ibm.geometry import Sphere

def test_sphere_surface_area():
    """Verify ∑ dS ≈ 4πR²."""
    radius = 0.5
    sphere = Sphere([0, 0, 0], radius=radius, n_points=2000)
    total_area = torch.sum(sphere.dS)
    expected = 4 * np.pi * radius**2
    # Should be exact since dS is defined as total_area / n_points
    assert torch.abs(total_area - expected) / expected < 1e-6

def test_sphere_point_distribution():
    """Verify points are on the sphere surface."""
    center = [1.0, 2.0, 3.0]
    radius = 1.5
    sphere = Sphere(center, radius, n_points=100)
    
    # Check distance from center
    center_tensor = torch.tensor(center, dtype=torch.float32)
    dist = torch.norm(sphere.x_lag - center_tensor, dim=1)
    
    # All points should be at distance R
    assert torch.allclose(dist, torch.tensor(radius), atol=1e-5)

def test_sphere_normals():
    """Verify normals are unit vectors pointing outward."""
    center = [0, 0, 0]
    radius = 1.0
    sphere = Sphere(center, radius, n_points=100)
    
    # Normals should be unit length
    norms = torch.norm(sphere.normals, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    # Normals should be parallel to position vector (for centered sphere)
    # x_lag = radius * normal
    assert torch.allclose(sphere.x_lag, radius * sphere.normals, atol=1e-5)

if __name__ == "__main__":
    test_sphere_surface_area()
    test_sphere_point_distribution()
    test_sphere_normals()
    print("All tests passed!")
