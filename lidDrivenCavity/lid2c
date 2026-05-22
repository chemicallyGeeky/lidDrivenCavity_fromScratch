#lid driven cavity with SIMPLE and PWIM, Re 10-75, 4x4-16x16 mesh, implicit relaxation, sparse matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

uLid = 0.2 # lid velocity = 1 cm/s
N = 64 # no of cells in each direction
len = 0.01 #each side of the sq. cavity
x = len/N #delta_x
y = len/N #delta_y
outer = 10000  # number of maximum outer loops

#constants
nu = 1e-5 # kinematic viscosity
Re = (len*uLid)/nu

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
            #CORNERS
            if i == 0 and j == 0:  #left-bot
                e = o + N
                n = o + 1
                uEqn[o, o] = (ue[i,j]*y + vn[i,j]*x + 6 *nu)
                uEqn[o, e] = - nu
                uEqn[o, n] = -nu
            elif i == 0 and j == (N-1): #left-top
                e = o + N
                s = o - 1
                uEqn[o, o] = (ue[i,j]*y + 6 *nu)  
                uEqn[o, e] = -nu
                uEqn[o, s] = -(vs[i,j]*x + nu)
            elif i == (N-1) and j == 0: #right-bot
                w = o - N
                n = o + 1
                uEqn[o, o] = (vn[i,j]*x + 6 *nu)  
                uEqn[o, w] = -(uw[i,j]*y + nu)
                uEqn[o, n] = -nu
            elif i == (N-1) and j == (N-1): #right-top
                o = N*i + j
                w = o - N
                s = o - 1
                uEqn[o , o] = 6*nu
                uEqn[o, w] = -(uw[i,j]*y + nu)
                uEqn[o, s] = -(vs[i,j]*x + nu)
            # SIDES    
            elif i == 0: #left
                e = o + N
                n = o + 1
                s = o - 1
                uEqn[o , o] = ue[i,j]*y + vn[i,j]*x + 5*nu
                uEqn[o, e] = -nu
                uEqn[o, n] = -nu
                uEqn[o, s] = -(vs[i,j]*x + nu)
            elif i == (N-1): #right
                o = N*i + j
                w = o - N
                n = o + 1
                s = o - 1
                uEqn[o , o] = vn[i,j]*x + 5*nu
                uEqn[o, w] = -(uw[i,j]*y + nu)
                uEqn[o, n] = -nu
                uEqn[o, s] = -(vs[i,j]*x + nu)
            elif j == 0: # bot
                e = o + N
                w = o - N
                n = o + 1
                uEqn[o , o] = ue[i,j]*y + vn[i,j]*x + 5*nu
                uEqn[o, e] = -nu
                uEqn[o, w] = -(uw[i,j]*y + nu)
                uEqn[o, n] = -nu
            elif j == (N-1): #top
                e = o + N
                w = o - N
                s = o - 1
                uEqn[o, o] = (ue[i,j]*y + 5*nu)
                uEqn[o, e] = -nu
                uEqn[o, w] =-(uw[i,j]*y + nu)
                uEqn[o, s] = -(vs[i,j]*x + nu)
            else: #inner
                w = o - N
                e = o + N
                s = o - 1
                n = o + 1
                uEqn[o, o] = (ue[i,j]*y + vn[i,j]*x + 4*nu)
                uEqn[o, e] = -nu
                uEqn[o, w] =-(uw[i,j]*y + nu)
                uEqn[o, n] = -nu
                uEqn[o, s] = -(vs[i,j]*x + nu)

         
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
        
    #solve 
    uEqn_sparse = sp.csc_matrix(uEqn)
    uM = spla.spsolve(uEqn_sparse, bx)
    vM = spla.spsolve(uEqn_sparse, by)

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
            # for i = 0, j = 0, the value is pinned
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
    #solve
    pEqn_sparse = sp.csc_matrix(pEqn)
    pPrime = spla.spsolve(pEqn_sparse, mIm)
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
    if (u_res < tol) and (v_res < tol) and (p_res < tol) and (massImbalance < tol):
        break

breakpoint()

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
#plot - u
# Create cell center coordinates for the plot boundaries
x_centers = np.linspace(x / 2, len - x / 2, N)
y_centers = np.linspace(y / 2, len - y / 2, N)

plt.figure(figsize=(6, 5))

# u.T aligns your [x, y] indexing with imshow's [row, col] requirement
# origin='lower' ensures j=0 starts at the bottom
im = plt.imshow(u.T, origin='lower', extent=[0, len, 0, len], cmap='jet')

# Add decorations
plt.colorbar(im, label='X velocity (m/s)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Lid-Driven Cavity Velocity (Re = {Re}, Grid = {N}x{N})')

#plt.tight_layout()
plt.show()

#plot - v
# Create cell center coordinates for the plot boundaries
plt.figure(figsize=(6, 5))

# v.T aligns your [x, y] indexing with imshow's [row, col] requirement
# origin='lower' ensures j=0 starts at the bottom
im = plt.imshow(v.T, origin='lower', extent=[0, len, 0, len], cmap='jet')

# Add decorations
plt.colorbar(im, label='Y velocity (m/s)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Lid-Driven Cavity Velocity (Re = {Re}, Grid = {N}x{N})')

plt.tight_layout()
plt.show()

#plot - p
# Create cell center coordinates for the plot boundaries
plt.figure(figsize=(6, 5))

# p.T aligns your [x, y] indexing with imshow's [row, col] requirement
# origin='lower' ensures j=0 starts at the bottom
im = plt.imshow(p.T, origin='lower', extent=[0, len, 0, len], cmap='jet')

# Add decorations
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
# Add decorations matching your style
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(0, len)
plt.ylim(0, len)
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
# Add decorations matching your style
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(0, len)
plt.ylim(0, len)
plt.title(f'Velocity Vectors (Re = {Re}, Grid = {N}x{N})')
plt.show()

breakpoint()

