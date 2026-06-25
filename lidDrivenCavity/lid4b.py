#lid driven cavity with SIMPLE and PWIM, implicit relaxation, dense matrix
import numpy as np
import matplotlib.pyplot as plt

uLid = 0.1 # lid velocity = 1 cm/s
N = 32 # no of cells in each direction
L = 0.01 #each side of the sq. cavity
x = L/N #delta_x
y = L/N #delta_y
outer = 10000  # number of maximum outer loops

#constants
nu = 1e-5 # kinematic viscosity
Re = (L*uLid)/nu

u_rel = 0.7
p_rel = 0.3
tol = 1e-6

#create arrays and initialize
u = np.zeros([N, N])
v = np.zeros([N, N])
p = np.zeros([N, N]) # P/density

#links coefficients
def calculate_u_flux(u):
    #eastern face flux
    ue = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i == (N-1):
                ue[i, j] = 0
            else:
                ue[i, j] = 0.5*(u[i,j] + u[i+1, j])  
    #western face flux
    uw = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i == 0:
                uw[i, j] = 0
            else:
                uw[i, j] = 0.5*(u[i,j] + u[i-1, j])    
    return ue, uw             
def calculate_v_flux(v):
    #northern face flux
    vn = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if j == (N-1):
                vn[i, j] = 0
            else:
                vn[i, j] = 0.5*(v[i,j] + v[i, j+1])   
    #southern face flux
    vs = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if j == 0:
                vs[i, j] = 0
            else:
                vs[i, j] = 0.5*(v[i,j] + v[i, j-1]) 
    return vn, vs            

#for plotting residuals
massImbalance_all = []
u_residual_all = []
v_residual_all = []
p_residual_all = []

for loop in range(outer):
    #STEP 1: solve the momentum eqn.
    ue, uw = calculate_u_flux(u)
    vn, vs = calculate_v_flux(v)
    M = N*N
    uEqn = np.zeros([M, M])
    for i in range(0, N):
        for j in range(0, N):
            o = N*i + j
            #the fluxes
            Fe = ue[i,j]*y
            Fw = uw[i,j]*y
            Fn = vn[i,j]*x
            Fs = vs[i,j]*x
            #CORNERS
            if i == 0 and j == 0:  #left-bot
                e = o + N
                n = o + 1
                uEqn[o, o] = max(0, Fe) + max(0, Fn) + 6 *nu
                uEqn[o, e] = - (max(0, -Fe) + nu)
                uEqn[o, n] = -(max(0, -Fn) + nu)
            elif i == 0 and j == (N-1): #left-top
                e = o + N
                s = o - 1
                uEqn[o, o] = (max(0, Fe) + max(0, -Fs) + 6 *nu)  
                uEqn[o, e] = -(max(0, -Fe) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)
            elif i == (N-1) and j == 0: #right-bot
                w = o - N
                n = o + 1
                uEqn[o, o] = (max(0, Fn) + max(0, -Fw) + 6 *nu)  
                uEqn[o, w] = -(max(0, Fw) + nu)
                uEqn[o, n] = -(max(0, -Fn) + nu)
            elif i == (N-1) and j == (N-1): #right-top
                o = N*i + j
                w = o - N
                s = o - 1
                uEqn[o , o] = max(0, -Fw) + max(0, -Fs) + 6*nu
                uEqn[o, w] = -(max(0, Fw) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)
            # SIDES    
            elif i == 0: #left
                e = o + N
                n = o + 1
                s = o - 1
                uEqn[o , o] = max(0, Fe) + max(0, Fn) + max(0, -Fs) + 5*nu
                uEqn[o, e] = -(max(0, -Fe) + nu)
                uEqn[o, n] = -(max(0, -Fn) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)
            elif i == (N-1): #right
                o = N*i + j
                w = o - N
                n = o + 1
                s = o - 1
                uEqn[o , o] = max(0, -Fw) + max(0, Fn) + max(0, -Fs) + 5*nu
                uEqn[o, w] = -(max(0, Fw) + nu)
                uEqn[o, n] = -(max(0, -Fn) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)
            elif j == 0: # bot
                e = o + N
                w = o - N
                n = o + 1
                uEqn[o , o] = max(0, Fe) + max(0, -Fw) + max(0, Fn) + 5*nu
                uEqn[o, e] = -(max(0, -Fe) + nu)
                uEqn[o, w] = -(max(0, Fw) + nu)
                uEqn[o, n] = -(max(0, -Fn)+ nu)
            elif j == (N-1): #top
                e = o + N
                w = o - N
                s = o - 1
                uEqn[o, o] = max(0, Fe) + max(0, -Fw) + max(0, -Fs) + 5*nu
                uEqn[o, e] = -(max(0, -Fe) + nu)
                uEqn[o, w] =-(max(0, Fw) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)
            else: #inner
                w = o - N
                e = o + N
                s = o - 1
                n = o + 1
                uEqn[o, o] = max(0, Fe) + max(0, -Fw) + max(0, Fn) + max(0, -Fs) + 4*nu
                uEqn[o, e] = -(max(0, -Fe) + nu)
                uEqn[o, w] = -(max(0, Fw) + nu)
                uEqn[o, n] = -(max(0, -Fn) + nu)
                uEqn[o, s] = -(max(0, Fs) + nu)

        
    #bx
    bx = np.zeros(M)
    for i in range(N):
        for j in range(N):
            o = N*i + j
            if i == 0 and j == (N-1): #top-left
                bx[o] = 0.5*(p[i, j] - p[i+1, j])*y + 2*nu*uLid
            elif i == (N-1) and j == (N-1): #top-right
                bx[o] = 0.5*(p[i-1, j] - p[i, j])*y + 2*nu*uLid
            elif j == (N-1): #top-inner
                bx[o] = 0.5*(p[i-1, j] - p[i+1, j])*y + 2*nu*uLid    
            elif i == 0: #left cells
                bx[o] = 0.5*(p[i, j] - p[i+1, j])*y
            elif i == (N-1): #right cells
                bx[o] = 0.5*(p[i-1, j] - p[i, j])*y
            else: #inner
                bx[o] = 0.5*(p[i-1, j] - p[i+1, j])*y

    #by
    by = np.zeros(M)
    for i in range(N):
        for j in range(N):
            o = N*i + j
            if j == 0: #bot cells
                by[o] = 0.5*(p[i, j] - p[i, j+1])*x
            elif j == (N-1): #top cells
                by[o] = 0.5*(p[i, j-1] - p[i, j])*x
            else:
                by[o] = 0.5*(p[i, j-1] - p[i, j+1])*x       

    #modify for implicit relaxation
    for k in range(M):
        uEqn[k, k] = uEqn[k, k]/u_rel
        bx[k] = bx[k] + uEqn[k,k]*(1-u_rel)*u.reshape(-1)[k]
        by[k] = by[k] + uEqn[k,k]*(1-u_rel)*v.reshape(-1)[k]
        
    #solve directly
    uM = np.linalg.solve(uEqn, bx) # Mx1 vector
    vM = np.linalg.solve(uEqn, by) # Mx1 vector
    uHat = uM.reshape(N, N)
    vHat = vM.reshape(N, N)
    #breakpoint()

    #STEP 2: solve the pressure correction equation
    #recalculate fluxes with PWIM
    #mass imbalance in pressure correction equation (RHS)
    ueHat, uwHat = calculate_u_flux(uHat)
    vnHat, vsHat = calculate_v_flux(vHat)
    mIm = np.zeros(M)
    for i in range(N):
        for j in range(N):
            o = N*i + j
            # for i = 0, j = 0, the values is pinned
            if o > 0: 
                mIm[o] = -((ueHat[i,j] - uwHat[i,j]) + (vnHat[i,j] -  vsHat[i,j]))

    #storing the coefficients of the LHS
    a0 = np.diag(uEqn)
    #w: o - N
    ucf_w = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i > 0:
                o = N*i + j
                w = o - N
                ucf_w[i, j] = 0.5*(1/a0[w] + 1/a0[o])

    #e: o + N
    ucf_e = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i < N-1:
                o = N*i + j
                e = o + N
                ucf_e[i, j] = 0.5*(1/a0[e] + 1/a0[o])

    #s: o - 1
    vcf_s = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if j > 0:
                o = N*i + j
                s = o - 1
                vcf_s[i, j] = 0.5*(1/a0[s] + 1/a0[o])      

    #n: o+1
    vcf_n = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if j < N-1:
                o = N*i + j
                n = o + 1
                vcf_n[i, j] = 0.5*(1/a0[n] + 1/a0[o]) 

    #assembling the pressure-correction matrix
    pEqn = np.zeros([M, M])
    for i in range(0, N):
        for j in range(0, N):
            o = N*i + j
            #CORNERS
            if i == 0 and j == 0:  #left-bot, pinned
                e = o + N
                n = o + 1
                pEqn[o, o] = 1
            elif i == 0 and j == (N-1): #left-top
                e = o + N
                s = o - 1
                pEqn[o, o] = ucf_e[i,j] + vcf_s[i, j] 
                pEqn[o, e] = -ucf_e[i,j]
                pEqn[o, s] = -vcf_s[i, j]
            elif i == (N-1) and j == 0: #right-bot
                w = o - N
                n = o + 1
                pEqn[o, o] =  ucf_w[i,j] + vcf_n[i, j] 
                pEqn[o, w] = -ucf_w[i,j]
                pEqn[o, n] = -vcf_n[i, j]
            elif i == (N-1) and j == (N-1): #right-top
                o = N*i + j
                w = o - N
                s = o - 1
                pEqn[o , o] = ucf_w[i,j] + vcf_s[i, j] 
                pEqn[o, w] = -ucf_w[i,j]
                pEqn[o, s] = -vcf_s[i, j] 
            # SIDES    
            elif i == 0: #left
                o = N*i + j
                e = o + N
                n = o + 1
                s = o - 1
                pEqn[o , o] = ucf_e[i,j] + vcf_n[i, j] + vcf_s[i, j]
                pEqn[o, e] = -ucf_e[i,j]
                pEqn[o, n] = -vcf_n[i, j]
                pEqn[o, s] = -vcf_s[i, j]
            elif i == (N-1): #right
                o = N*i + j
                w = o - N
                n = o + 1
                s = o - 1
                pEqn[o , o] = ucf_w[i,j] + vcf_n[i, j] + vcf_s[i, j]
                pEqn[o, w] = -ucf_w[i,j]
                pEqn[o, n] = -vcf_n[i, j]
                pEqn[o, s] = -vcf_s[i, j]
            elif j == 0: # bot
                o = N*i + j
                e = o + N
                w = o - N
                n = o + 1
                pEqn[o , o] = ucf_e[i,j] + ucf_w[i,j] + vcf_n[i, j] 
                pEqn[o, e] = -ucf_e[i,j]
                pEqn[o, w] = -ucf_w[i,j]
                pEqn[o, n] = -vcf_n[i, j]
            elif j == (N-1): #top
                o = N*i + j
                e = o + N
                w = o - N
                s = o - 1
                pEqn[o, o] = ucf_e[i,j] + ucf_w[i,j] + vcf_s[i, j] 
                pEqn[o, e] = -ucf_e[i,j]
                pEqn[o, w] =-ucf_w[i,j]
                pEqn[o, s] = -vcf_s[i, j]
            else:
                w = o - N
                e = o + N
                s = o - 1
                n = o + 1
                pEqn[o, o] = ucf_e[i,j] + ucf_w[i,j] + vcf_n[i, j] + vcf_s[i, j] 
                pEqn[o, e] = -ucf_e[i,j]
                pEqn[o, w] =-ucf_w[i,j]
                pEqn[o, n] = -vcf_n[i, j]
                pEqn[o, s] = -vcf_s[i, j]

    massImbalance = np.abs(np.sum(mIm))
    print(massImbalance)
    massImbalance_all.append(massImbalance)
    pEqn = pEqn*x
    pPrime = np.linalg.solve(pEqn, mIm)
    pPrime = pPrime.reshape(N, N)

    #STEP 3: updating the values
    #p
    p_new = p + p_rel*pPrime

    #u, v
    a0 = a0.reshape(N, N)
    uPrime = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i == 0:
                uPrime[i, j] = (pPrime[i,j] - pPrime[i+1, j])/(2*a0[i,j])*y 
            elif i == (N-1):
                uPrime[i,j] =  (pPrime[i-1,j] - pPrime[i, j])/(2*a0[i,j])*y 
            else:
                uPrime[i, j] = (pPrime[i-1,j] - pPrime[i+1, j])/(2*a0[i,j])*y    

    vPrime = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if j == 0:
                vPrime[i, j] = (pPrime[i,j] - pPrime[i, j+1])/(2*a0[i,j])*x 
            elif j == (N-1):
                vPrime[i,j] =  (pPrime[i,j-1] - pPrime[i, j])/(2*a0[i,j])*x 
            else:
                vPrime[i, j] = (pPrime[i,j-1] - pPrime[i, j+1])/(2*a0[i,j])*x                           
                    
    u_new = uHat + uPrime
    v_new = vHat + vPrime    
        
    #residuals
    u_res = np.max(np.abs(u_new - u))
    v_res = np.max(np.abs(v_new - v))
    p_res = np.max(np.abs(p_new - p))
    u_residual_all.append(u_res)
    v_residual_all.append(v_res)
    p_residual_all.append(p_res)

    #next outer loop
    u = u_new
    v = v_new
    p = p_new

    #check for convergence
    if (u_res < tol) and (v_res < tol) and (p_res < tol) and (massImbalance < 1e-6):
        break

breakpoint()

#PLOTS
#plot residuals
iter = np.arange(loop+1)
plt.figure(figsize=(8, 4))
plt.plot(iter, massImbalance_all, '*', color='k', markersize=1); plt.xlabel('iteration #'); plt.ylabel('residual')
plt.title('mass imbalance'); plt.show()
plt.figure(figsize=(8, 4))
plt.plot(iter, u_residual_all, '*', color='k', markersize=1); plt.xlabel('iteration #'); plt.ylabel('residual')
plt.title('u residuals'); plt.show()
plt.figure(figsize=(8, 4))
plt.plot(iter, v_residual_all, '*', color='k', markersize=1); plt.xlabel('iteration #'); plt.ylabel('residual')
plt.title('v residuals'); plt.show()
plt.figure(figsize=(8, 4))
plt.plot(iter,p_residual_all, '*', color='k', markersize=1); plt.xlabel('iteration #'); plt.ylabel('residual')
plt.title('p residuals'); plt.show()

#printing arrays: left and right are correct, top and bottom are reversed
#np.flipud(u.T)
#np.flipud(v.T)
#np.flipud(p.T)

#plot: colormaps
# Create cell center coordinates for the plot boundaries
x_centers = np.linspace(x / 2, L - x / 2, N)
y_centers = np.linspace(y / 2, L - y / 2, N)

#plot - u
plt.figure(figsize=(6, 5))
# u.T aligns your [x, y] indexing with imshow's [row, col] requirement
# origin='lower' ensures j=0 starts at the bottom
im = plt.imshow(u.T, origin='lower', extent=[0, L, 0, L], cmap='jet')
plt.colorbar(im, label='X velocity (m/s)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Lid-Driven Cavity Velocity (Re = {Re}, Grid = {N}x{N})')
plt.show()

#plot - v
plt.figure(figsize=(6, 5))
im = plt.imshow(v.T, origin='lower', extent=[0, L, 0, L], cmap='jet')
plt.colorbar(im, label='Y velocity (m/s)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Lid-Driven Cavity Velocity (Re = {Re}, Grid = {N}x{N})')
plt.tight_layout()
plt.show()

#plot - p
plt.figure(figsize=(6, 5))
# p.T aligns your [x, y] indexing with imshow's [row, col] requirement
# origin='lower' ensures j=0 starts at the bottom
im = plt.imshow(p.T, origin='lower', extent=[0, L, 0, L], cmap='jet')
plt.colorbar(im, label='Kinematic pressure (m2/s2)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Lid-Driven Cavity Velocity (Re = {Re}, Grid = {N}x{N})')
plt.tight_layout()
plt.show()

#streamline plot
# 1. Create the cell center coordinates and meshgrid
X, Y = np.meshgrid(x_centers, y_centers)
# 2. Calculate velocity magnitude matching your transposed layout
vel_magnitude = np.sqrt(u.T**2 + v.T**2)
plt.figure(figsize=(6.5, 5))
# 3. Plot the color background using a filled contour map of the velocity magnitude
im = plt.contourf(X, Y, vel_magnitude, levels=50, cmap='jet')
cbar = plt.colorbar(im, label='Velocity Magnitude (m/s)')
# 4. Overlay the black streamlines to show the vortex paths clearly
plt.streamplot(X, Y, u.T, v.T, color='black', linewidth=1.2, density=1)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(0, L)
plt.ylim(0, L)
plt.title(f'Velocity Fields & Streamlines (Re = {Re}, Grid = {N}x{N})')
plt.show()

#quiver plot
plt.figure(figsize=(6, 5))
# Removing scale_units='xy' forces matplotlib to auto-scale arrow lengths visibly
q = plt.quiver(X, Y, u.T, v.T, angles='xy', color='black')
# 2. Add the Quiver Key (Scale Bar)
# labelpos='E' puts the text to the East (right) of the arrow
# coordinates='axes' places it relative to the plot frame (1.05 is slightly outside to the right)
#plt.quiverkey(q, X=0.85, Y=1.05, U=1, label='1 m/s', labelpos='E', coordinates='axes')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(0, L)
plt.ylim(0, L)
plt.title(f'Velocity Vectors (Re = {Re}, Grid = {N}x{N})')
plt.show()

breakpoint()

#line profiles
def line_profiles(u, v, N, L=0.01, U_lid=0.1):
    # Non-dimensionalized coordinates (0 to 1)
    normalized_centers = np.linspace(1/(2*N), 1 - 1/(2*N), N)
    mid_idx = N // 2
    
    # Vertical centerline (x = 0.5): Fix X-index (i), extract all Y-indices (j)
    u_profile = u[mid_idx, :] / U_lid
    
    # Horizontal centerline (y = 0.5): Fix Y-index (j), extract all X-indices (i)
    v_profile = v[:, mid_idx] / U_lid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: u-velocity along vertical centerline
    ax1.plot(u_profile, normalized_centers, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax1.set_title('Normalized u-velocity (Vertical Centerline)')
    ax1.set_xlabel('$u / U_{lid}$')
    ax1.set_ylabel('$y / L$')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right: v-velocity along horizontal centerline
    ax2.plot(normalized_centers, v_profile, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax2.set_title('Normalized v-velocity (Horizontal Centerline)')
    ax2.set_xlabel('$x / L$')
    ax2.set_ylabel('$v / U_{lid}$')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

line_profiles(u, v, N, L=0.01, U_lid=0.1)

#comparison with Ghia et. al. for Re 100

def plot_ghia_comparison(u, v, N, L=0.01, U_lid=0.1):
    """
    Extracts, normalizes, and plots centerline profiles matching the 
    (i=X, j=Y) grid orientation against Ghia et al. (1982) at Re=100.
    """
    # -----------------------------------------------------------------
    # 1. Ghia et al. (1982) Benchmark Data (Re = 100)
    # -----------------------------------------------------------------
    #u velocity across vertical line in the middle
    y_ghia = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                       0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    u_ghia = np.array([1.0000, 0.8412, 0.7887, 0.7372, 0.68717, 0.2315, 0.0033, -0.1364, 
                       -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.04775, -0.0419, -0.0371, 0.0000])

    #v velocity across horizontal line in the middle
    x_ghia = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
    v_ghia = np.array([0.0000, -0.05906, -0.0739, -0.0886, -0.10313, -0.16914, -0.22445, -0.24533, 
                       0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.1089, 0.1009, 0.0923, 0.0000])

    # -----------------------------------------------------------------
    # 2. Extract & Normalize Centerlines (Accounting for i=X, j=Y)
    # -----------------------------------------------------------------
    # Non-dimensionalized coordinates (0 to 1)
    normalized_centers = np.linspace(1/(2*N), 1 - 1/(2*N), N)
    mid_idx = N // 2
    
    # Vertical centerline (x = 0.5): Fix X-index (i), extract all Y-indices (j)
    u_profile = u[mid_idx, :] / U_lid
    
    # Horizontal centerline (y = 0.5): Fix Y-index (j), extract all X-indices (i)
    v_profile = v[:, mid_idx] / U_lid

    # -----------------------------------------------------------------
    # 3. Plotting
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: u-velocity along vertical centerline
    ax1.plot(u_profile, normalized_centers, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax1.scatter(u_ghia, y_ghia, color='darkred', marker='o', facecolors='none', s=40, label='Ghia et al. (1982)')
    ax1.set_title('Normalized u-velocity (Vertical Centerline)')
    ax1.set_xlabel('$u / U_{lid}$')
    ax1.set_ylabel('$y / L$')
    ax1.set_xlim([-0.3, 1.1])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Right: v-velocity along horizontal centerline
    ax2.plot(normalized_centers, v_profile, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax2.scatter(x_ghia, v_ghia, color='darkred', marker='o', facecolors='none', s=40, label='Ghia et al. (1982)')
    ax2.set_title('Normalized v-velocity (Horizontal Centerline)')
    ax2.set_xlabel('$x / L$')
    ax2.set_ylabel('$v / U_{lid}$')
    ax2.set_ylim([-0.3, 0.3])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if Re == 100.0:
    plot_ghia_comparison(u, v, N)    

breakpoint()

def plot_ghia_comparison_re1000(u, v, N, L=0.01, U_lid=1.0):
    """
    Extracts, normalizes, and plots centerline profiles matching the 
    (i=X, j=Y) grid orientation against Ghia et al. (1982) at Re=1000.
    """
    # -----------------------------------------------------------------
    # 1. Ghia et al. (1982) Benchmark Data (Re = 1000)
    # -----------------------------------------------------------------
    y_ghia = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                       0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    u_ghia = np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, 
                       -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000])

    x_ghia = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                       0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
    v_ghia = np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51500, -0.42665, -0.31966, 
                       0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000])

    # -----------------------------------------------------------------
    # 2. Extract & Normalize Centerlines (Accounting for i=X, j=Y)
    # -----------------------------------------------------------------
    normalized_centers = np.linspace(1/(2*N), 1 - 1/(2*N), N)
    mid_idx = N // 2
    
    # Vertical centerline (x = 0.5): Fix X-index (i), extract all Y-indices (j)
    u_profile = u[mid_idx, :] / U_lid
    
    # Horizontal centerline (y = 0.5): Fix Y-index (j), extract all X-indices (i)
    v_profile = v[:, mid_idx] / U_lid

    # -----------------------------------------------------------------
    # 3. Plotting
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: u-velocity along vertical centerline
    ax1.plot(u_profile, normalized_centers, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax1.scatter(u_ghia, y_ghia, color='darkred', marker='o', facecolors='none', s=40, label='Ghia et al. (1982) Re=1000')
    ax1.set_title('Normalized u-velocity (Vertical Centerline - Re=1000)')
    ax1.set_xlabel('$u / U_{lid}$')
    ax1.set_ylabel('$y / L$')
    ax1.set_xlim([-0.6, 1.1])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Right: v-velocity along horizontal centerline
    ax2.plot(normalized_centers, v_profile, 'b-', label='Current Solver (Normalized)', linewidth=2)
    ax2.scatter(x_ghia, v_ghia, color='darkred', marker='o', facecolors='none', s=40, label='Ghia et al. (1982) Re=1000')
    ax2.set_title('Normalized v-velocity (Horizontal Centerline - Re=1000)')
    ax2.set_xlabel('$x / L$')
    ax2.set_ylabel('$v / U_{lid}$')
    ax2.set_ylim([-0.6, 0.6])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if Re == 1000.0:
    plot_ghia_comparison_re1000(u, v, N)    

breakpoint()    
