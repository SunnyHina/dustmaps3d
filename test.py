from dustmaps3d import dustmaps3d
EBV, dust, sigma, max_d = dustmaps3d(1, 2, 1)

print("E(B-V):", EBV)
print("Dust density:", dust)
print("Sigma:", sigma)
print("Max distance per pixel:", max_d)
