import torch

# Test the reshape/permute logic used in solve_poisson
nx, ny, nz = 8, 8, 64

# Create a test array with known values
# Put a unique value at each (i,j,k) position: value = i*1000 + j*100 + k
test_data = torch.zeros(nx, ny, nz)
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            test_data[i, j, k] = i*1000 + j*100 + k

print("Original data shape:", test_data.shape)
print("Test value at (0,0,0):", test_data[0,0,0].item(), "should be 0")
print("Test value at (1,0,0):", test_data[1,0,0].item(), "should be 1000")
print("Test value at (0,1,0):", test_data[0,1,0].item(), "should be 100")
print("Test value at (0,0,1):", test_data[0,0,1].item(), "should be 1")
print("Test value at (2,3,5):", test_data[2,3,5].item(), "should be 2305")

# Apply the permute and reshape as in solve_poisson
# Permute to match matrix indexing: i varies fastest
data_permuted = test_data.permute(2, 1, 0)  # (nx,ny,nz) -> (nz,ny,nx)
data_flat = data_permuted.reshape(-1)

print("\nAfter permute(2,1,0) and reshape:")
print("Flat array length:", len(data_flat), "should be", nx*ny*nz)

# Check if the flat indexing matches matrix indexing: idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
# For i=1, j=1, k=1 (0-indexed: i=0, j=0, k=0)
idx_000 = 0 + 0*nx + 0*nx*ny
print(f"\nFlat index for (i=0,j=0,k=0): {idx_000}, value: {data_flat[idx_000].item()}, should be 0")

# For i=1 (idx 0), j=0, k=0
idx_100 = 1 + 0*nx + 0*nx*ny
print(f"Flat index for (i=1,j=0,k=0): {idx_100}, value: {data_flat[idx_100].item()}, should be 1000")

# For i=0, j=1, k=0
idx_010 = 0 + 1*nx + 0*nx*ny
print(f"Flat index for (i=0,j=1,k=0): {idx_010}, value: {data_flat[idx_010].item()}, should be 100")

# For i=0, j=0, k=1  
idx_001 = 0 + 0*nx + 1*nx*ny
print(f"Flat index for (i=0,j=0,k=1): {idx_001}, value: {data_flat[idx_001].item()}, should be 1")

# For i=2, j=3, k=5
idx_235 = 2 + 3*nx + 5*nx*ny
print(f"Flat index for (i=2,j=3,k=5): {idx_235}, value: {data_flat[idx_235].item()}, should be 2305")

# Now reverse: reshape and permute back
data_reshaped = data_flat.reshape(nz, ny, nx)
data_back = data_reshaped.permute(2, 1, 0)  # (nz,ny,nx) -> (nx,ny,nz)

print("\nAfter reverse transformation:")
print("Shape:", data_back.shape, "should be", (nx, ny, nz))
print("Value at (0,0,0):", data_back[0,0,0].item(), "should be 0")
print("Value at (1,0,0):", data_back[1,0,0].item(), "should be 1000")
print("Value at (0,1,0):", data_back[0,1,0].item(), "should be 100")
print("Value at (0,0,1):", data_back[0,0,1].item(), "should be 1")
print("Value at (2,3,5):", data_back[2,3,5].item(), "should be 2305")

# Check if we get back the original
if torch.allclose(test_data, data_back):
    print("\n✓ SUCCESS: Reshape/permute is reversible!")
else:
    print("\n✗ FAILURE: Data was corrupted!")
    print("Max difference:", torch.max(torch.abs(test_data - data_back)).item())
