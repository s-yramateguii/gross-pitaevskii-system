import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

def spc_rhs(t, initial, nx, ny, nz, N, KX, KY, KZ, K):
    # Split the state vector
    psire = initial.reshape(nx, ny, nz)
    psi = ifftn(psire)
    
    rhs = 1j * (0.5 * (-K) * psire - fftn(np.abs(psi)**2 * psi) + 
                fftn((A1 * (np.sin(X)**2) + B1) * (A2 * (np.sin(Y)**2) + B2) * 
                     (A3 * (np.sin(Z)**2) + B3) * psi))
    return rhs.flatten()

tspan = np.arange(0, 4.5, 0.5)
t_eval = tspan  # Times at which to store solution
t_range = (tspan[0], tspan[-1])
Lx, Ly, Lz = 2*np.pi, 2*np.pi, 2*np.pi
nx, ny, nz = 16, 16, 16
N = nx * ny * nz
A1, A2, A3 = -1, -1, -1
B1, B2, B3 = 1, 1, 1

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
z2 = np.linspace(-Lz/2, Lz/2, nz + 1)
z = z2[:nz]
X, Y, Z = np.meshgrid(x, y, z)
V = (A1 * np.sin(X)**2 + B1) * (A2 * np.sin(Y)**2 + B2) * (A3 * np.sin(Z)**2 + B3)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
kz = (2 * np.pi / Lz) * np.concatenate((np.arange(0, nz/2), np.arange(-nz/2, 0)))
kz[0] = 1e-6
KX, KY, KZ = np.meshgrid(kx, ky, kz)
K = KX**2 + KY**2 + KZ**2

'''
Part a
'''
psi = np.cos(X) * np.cos(Y) * np.cos(z)
psi0 = fftn(psi).flatten()
psi1 = solve_ivp(spc_rhs, t_range,  psi0, t_eval=t_eval, args=(nx, ny, nz, N, KX, KY, KZ, K))


'''
Part b
'''
psi = np.sin(X) * np.sin(Y) * np.sin(z)
psi0 = fftn(psi).flatten()
psi2 = solve_ivp(spc_rhs, t_range,  psi0, t_eval=t_eval, args=(nx, ny, nz, N, KX, KY, KZ, K))

'''
Isosurface Plots
'''
# Visualization using Isosurface Plot
# psi_final = np.fft.ifftn(psi1.y[:, -1].reshape(nx, ny, nz))

# # Only use the real part or magnitude of psi
# # Only use the real part of psi
# psi_final_real = np.real(psi_final)

# # Define an isosurface level, e.g., 0.1
# isosurface_level = 0.1

# # Create the 3D isosurface plot using plotly
# fig = go.Figure(data=go.Isosurface(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=psi_final_real.flatten(),  # Use real part of psi
#     isomin=isosurface_level,
#     isomax=psi_final_real.max(),
#     opacity=0.8,  # Adjust transparency for visibility
#     surface_count=1,  # Number of isosurfaces to plot
#     colorscale='Viridis'  # Color scale for visualization
# ))

# fig.update_layout(title='Isosurface of Real Part of Wave Function at t = ' + str(tspan[-1]),
#                   scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
# fig.show()

'''
Isosurface Animation
'''
# Function to generate an isosurface for the given time step
def generate_isosurface_frame(psi_final_real, X, Y, Z, isosurface_level):
    return go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=psi_final_real.flatten(),  # Use real part of psi
        isomin=isosurface_level,
        isomax=psi_final_real.max(),
        opacity=0.8,  # Adjust transparency for visibility
        surface_count=1,  # Number of isosurfaces to plot
        colorscale='Viridis'  # Color scale for visualization
    )

# Define the initial wave function
isosurface_level = 0.1

# Create the initial frame (first time step)
psi_final = np.fft.ifftn(psi1.y[:, 0].reshape(nx, ny, nz))  # Extract the solution at t=0
psi_final_real = np.real(psi_final)  # Only use the real part

# Create the initial figure
fig = go.Figure(
    data=[generate_isosurface_frame(psi_final_real, X, Y, Z, isosurface_level)],
    layout=go.Layout(
        title={
            'text': "Isosurface for Cosine Initial Condition",
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
            method='animate', args=[None, dict(frame=dict(duration=100, redraw=True),
            fromcurrent=True, mode='immediate', transition=dict(duration=200))])])],
        annotations=[dict(
            x=0.5, y=1.05,  # Position the time text above the plot
            xref="paper", yref="paper",
            text="Time: 0",  # Initial time
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center"
        )]
    ),
    frames=[go.Frame(
        data=[generate_isosurface_frame(np.real(np.fft.ifftn(psi1.y[:, i].reshape(nx, ny, nz))), X, Y, Z, isosurface_level)],
        name=str(i),
        layout=dict(
            annotations=[dict(
                x=0.5, y=1.05,  # Position the time text above the plot
                xref="paper", yref="paper",
                text=f"Time: {t_eval[i]}",  # Update the time text dynamically
                showarrow=False,
                font=dict(size=16, color="black"),
                align="center"
            )]
        )
    ) for i in range(len(t_eval))]  # Add frames for each time step
)

# Create smoother transitions by adding intermediate frames (reduce frame count for faster animation)
frame_count = 10  # Fewer intermediate frames for a faster animation
frames = []

# Add frames for the forward direction
for i in range(len(t_eval) - 1):
    start_idx = i
    end_idx = i + 1
    psi_start = np.fft.ifftn(psi1.y[:, start_idx].reshape(nx, ny, nz))
    psi_end = np.fft.ifftn(psi1.y[:, end_idx].reshape(nx, ny, nz))
    
    # Interpolate between frames
    for j in range(frame_count):
        t = j / frame_count  # Interpolation factor
        psi_interpolated = (1 - t) * psi_start + t * psi_end
        psi_interpolated_real = np.real(psi_interpolated)
        
        frames.append(go.Frame(
            data=[generate_isosurface_frame(psi_interpolated_real, X, Y, Z, isosurface_level)],
            name=f"{start_idx}-{end_idx}-{j}",
            layout=dict(
                annotations=[dict(
                    x=0.5, y=1.05,  # Position the time text above the plot
                    xref="paper", yref="paper",
                    text=f"Time: {t_eval[i] + t * (t_eval[i + 1] - t_eval[i])}",  # Interpolated time
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    align="center"
                )]
            )
        ))

# Add the final frame (to show the last time step)
final_frame = go.Frame(
    data=[generate_isosurface_frame(np.real(np.fft.ifftn(psi1.y[:, -1].reshape(nx, ny, nz))), X, Y, Z, isosurface_level)],
    name="final",
    layout=dict(
        annotations=[dict(
            x=0.5, y=1.05,  # Position the time text above the plot
            xref="paper", yref="paper",
            text=f"Time: {t_eval[-1]}",  # Final time
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center"
        )]
    )
)
frames.append(final_frame)

# Update frames and animation properties
fig.frames = frames

# Save the first animation
# fig.write_html("cos_initial_conditions_animation.html")
# Show the animation
fig.show()

# Sin Initial Conditions
isosurface_level = 0.1

# Create the initial frame (first time step)
psi_final = np.fft.ifftn(psi2.y[:, 0].reshape(nx, ny, nz))  # Extract the solution at t=0
psi_final_real = np.real(psi_final)  # Only use the real part

# Create the initial figure
fig = go.Figure(
    data=[generate_isosurface_frame(psi_final_real, X, Y, Z, isosurface_level)],
    layout=go.Layout(
        title={
            'text': "Isosurface for Sine Initial Condition",
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top'
        },
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
            method='animate', args=[None, dict(frame=dict(duration=100, redraw=True),
            fromcurrent=True, mode='immediate', transition=dict(duration=200))])])],
        annotations=[dict(
            x=0.5, y=1.05,  # Position the time text above the plot
            xref="paper", yref="paper",
            text="Time: 0",  # Initial time
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center"
        )]
    ),
    frames=[go.Frame(
        data=[generate_isosurface_frame(np.real(np.fft.ifftn(psi2.y[:, i].reshape(nx, ny, nz))), X, Y, Z, isosurface_level)],
        name=str(i),
        layout=dict(
            annotations=[dict(
                x=0.5, y=1.05,  # Position the time text above the plot
                xref="paper", yref="paper",
                text=f"Time: {t_eval[i]}",  # Update the time text dynamically
                showarrow=False,
                font=dict(size=16, color="black"),
                align="center"
            )]
        )
    ) for i in range(len(t_eval))]  # Add frames for each time step
)

# Create smoother transitions by adding intermediate frames (reduce frame count for faster animation)
frame_count = 10  # Fewer intermediate frames for a faster animation
frames = []

# Add frames for the forward direction
for i in range(len(t_eval) - 1):
    start_idx = i
    end_idx = i + 1
    psi_start = np.fft.ifftn(psi2.y[:, start_idx].reshape(nx, ny, nz))
    psi_end = np.fft.ifftn(psi2.y[:, end_idx].reshape(nx, ny, nz))
    
    # Interpolate between frames
    for j in range(frame_count):
        t = j / frame_count  # Interpolation factor
        psi_interpolated = (1 - t) * psi_start + t * psi_end
        psi_interpolated_real = np.real(psi_interpolated)
        
        frames.append(go.Frame(
            data=[generate_isosurface_frame(psi_interpolated_real, X, Y, Z, isosurface_level)],
            name=f"{start_idx}-{end_idx}-{j}",
            layout=dict(
                annotations=[dict(
                    x=0.5, y=1.05,  # Position the time text above the plot
                    xref="paper", yref="paper",
                    text=f"Time: {t_eval[i] + t * (t_eval[i + 1] - t_eval[i])}",  # Interpolated time
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    align="center"
                )]
            )
        ))

# Add the final frame (to show the last time step)
final_frame = go.Frame(
    data=[generate_isosurface_frame(np.real(np.fft.ifftn(psi2.y[:, -1].reshape(nx, ny, nz))), X, Y, Z, isosurface_level)],
    name="final",
    layout=dict(
        annotations=[dict(
            x=0.5, y=1.05,  # Position the time text above the plot
            xref="paper", yref="paper",
            text=f"Time: {t_eval[-1]}",  # Final time
            showarrow=False,
            font=dict(size=16, color="black"),
            align="center"
        )]
    )
)
frames.append(final_frame)

# Update frames and animation properties
fig.frames = frames

# Save the second animation
# fig.write_html("sin_initial_conditions_animation.html")

# Show the animation
fig.show()